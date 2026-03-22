"""
main.py — Horizon Bank CSR Query Summarizer API

Architecture:
  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
  │  index.html │───▶│   main.py    │───▶│    rag.py   │  (vector retrieval)
  │  (frontend) │    │  (FastAPI)   │───▶│ translator  │  (lang detect + translate)
  └─────────────┘    └──────────────┘    └─────────────┘

Request flow:
  1. Receive customer query (text and/or image)
  2. translator.detect_and_translate_input()  → English text + detected language
  3. rag.retrieve(english_text)               → relevant KB chunks
  4. LLM call with [KB chunks + translated query] → structured JSON analysis
  5. translator.translate_output_to_english() → English CSR-facing fields
  6. Return full result (original language + English translation)
"""

import os
import base64
import json
import re
import uuid
import warnings
import httpx
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─── Unicode font registration ────────────────────────────────────────────────
# Noto Sans covers 500+ languages/scripts. Place font files next to main.py.
# Download from: https://fonts.google.com/noto/specimen/Noto+Sans
def _register_unicode_fonts():
    font_dir = BASE_DIR  # same folder as main.py; change if fonts are elsewhere
    font_map = [
        ("NotoSans",        "NotoSans-Regular.ttf"),
        ("NotoSans-Bold",   "NotoSans-Bold.ttf"),
        ("NotoSans-Italic", "NotoSans-Italic.ttf"),
    ]
    registered = []
    for name, fname in font_map:
        path = font_dir / fname
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
            registered.append(name)
    return registered

# BASE_DIR must be defined before font registration
BASE_DIR = Path(__file__).parent

_REGISTERED_FONTS = _register_unicode_fonts()
_HAS_UNICODE_FONT = "NotoSans" in _REGISTERED_FONTS

# Font name helpers — falls back to Helvetica if Noto not installed
def _F(bold=False, italic=False):
    """Return best available font name."""
    if _HAS_UNICODE_FONT:
        if bold:   return "NotoSans-Bold"   if "NotoSans-Bold"   in _REGISTERED_FONTS else "NotoSans"
        if italic: return "NotoSans-Italic" if "NotoSans-Italic" in _REGISTERED_FONTS else "NotoSans"
        return "NotoSans"
    # fallback
    if bold:   return "Helvetica-Bold"
    if italic: return "Helvetica-Oblique"
    return "Helvetica"

from openai import OpenAI

# ── Import our modules ────────────────────────────────────────────────────────
from rag import retrieve as rag_retrieve, get_stats as rag_stats
from translator import (
    detect_and_translate_input,
    translate_output_to_english,
    translate_to_target_language,
    SUPPORTED_LANGUAGES,
    language_display,
)

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Horizon Bank — Query Summarizer API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)
app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")


# ─── Serve frontend ───────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse(str(BASE_DIR / "index.html"))


# ─── Model registry ───────────────────────────────────────────────────────────
VISION_MODEL = {
    "label":      "Llama 4 Scout Vision",
    "deployment": "meta-llama/llama-4-scout-17b-16e-instruct",
    "vision":     True,
}
TEXT_MODEL = {
    "label":      "Llama 3.3 70B",
    "deployment": "llama-3.3-70b-versatile",
    "vision":     False,
}

def select_model(has_image: bool) -> dict:
    return VISION_MODEL if has_image else TEXT_MODEL


# ─── OpenAI client ────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    api_key  = os.environ.get("API_KEY")
    endpoint = os.environ.get("API_ENDPOINT")

    if api_key:
        print(f"API_KEY: {api_key[:20]}...")
    else:
        print("WARNING: API_KEY MISSING")
    print(f"ENDPOINT: {endpoint}")

    if not api_key:
        raise HTTPException(status_code=500, detail="API_KEY not found in .env file")
    if not endpoint:
        raise HTTPException(status_code=500, detail="API_ENDPOINT not found in .env file")

    return OpenAI(
        api_key=api_key,
        base_url=endpoint,
        http_client=httpx.Client(verify=False),
    )


# ─── System prompt (uses RAG context, NOT full KB) ────────────────────────────
def build_system_prompt(rag_context: str) -> str:
    return f"""You are an expert AI assistant for Horizon Bank's customer service center.
You help Customer Service Representatives (CSRs) by analyzing customer queries and producing structured, actionable summaries.

CRITICAL SCOPE RULE:
- You ONLY handle banking-related queries: accounts, loans, cards, transactions, KYC, digital banking, fraud, statements, interest, EMI, etc.
- If the query is NOT related to banking or finance, return a scope error immediately.
- Out-of-scope examples: geography, general knowledge, cooking, sports, weather, coding, history, science.
- For out-of-scope, return ONLY:
{{"out_of_scope": true, "message": "I'm sorry, I can only assist with banking-related queries. Please contact Horizon Bank support at 1800-XXX-XXXX."}}

LANGUAGE INSTRUCTION:
- The input query has already been translated to English for you.
- Write the "summary" and "suggested_response_opening" fields in the DETECTED LANGUAGE specified in the user message.
- All other fields (query_type, urgency, action titles, etc.) must be in English.

KNOWLEDGE BASE (retrieved via RAG — use ONLY these figures, do not invent rates or policies):
{rag_context}

IMPORTANT — Use exact figures from the knowledge base:
- Credit card interest: calculate actual rupee amount (e.g. ₹27,600 × 3.49% = ₹962.64)
- EMI: state exact tenures (3/6/9/12/18/24 months) and exact rates
- Fees: quote exact amounts
- Fraud: cite the RBI 3-day zero-liability rule
- NEVER say "inform the customer of the rate" — STATE the actual rate

Return this JSON for in-scope queries:
{{
  "out_of_scope": false,
  "detected_language": "English",
  "query_title": "Short title max 8 words",
  "query_type": "Account Issue | Transaction Dispute | Card Services | Loan & Credit | KYC / Compliance | Digital Banking | Fraud & Security | Credit Card Billing | General Inquiry",
  "urgency": "critical|high|medium|low",
  "customer_sentiment": "frustrated|neutral|confused|angry|polite",
  "summary": "2-3 sentence summary in the customer's detected language",
  "key_issues": [{{"issue": "Label", "detail": "Specific detail with exact figures from KB"}}],
  "recommended_actions": [{{
    "action": "Action title (English)",
    "description": "Specific step-by-step instructions with exact rates/amounts from KB",
    "team": "Team name",
    "priority": "immediate|standard|low"
  }}],
  "escalation_required": false,
  "escalation_reason": "",
  "compliance_flags": [],
  "estimated_resolution_time": "e.g. 2-3 business days",
  "suggested_response_opening": "Professional empathetic opening in customer's detected language"
}}

Return ONLY the JSON. No markdown, no extra text."""


# ─── Core summarization function ──────────────────────────────────────────────
def summarize_query(
    query_text: Optional[str],
    image_b64:  Optional[str],
    image_mime: Optional[str],
) -> dict:
    """
    Full pipeline:
      1. Auto-select model
      2. Translate input → English
      3. RAG retrieve relevant KB chunks
      4. LLM analysis
      5. Translate output → English for CSR
      6. Return combined result
    """
    has_image = bool(image_b64)
    model     = select_model(has_image)
    client    = get_client()

    # ── Step 2: Language detection + translation ──────────────────────────────
    translation = None
    english_query = query_text or ""

    if query_text and query_text.strip():
        print("🌐 Detecting language and translating input...")
        translation = detect_and_translate_input(
            text       = query_text,
            client     = client,
            deployment = TEXT_MODEL["deployment"],  # always use text model for translation
        )
        english_query = translation.english_text
        print(f"   Language: {translation.language} | is_english: {translation.is_english}")

    # ── Step 3: RAG retrieval ─────────────────────────────────────────────────
    print("🔍 Running RAG retrieval...")
    rag_context = rag_retrieve(english_query or "banking customer query")
    print(f"   Retrieved context ({len(rag_context)} chars)")

    # ── Step 4: Build LLM content ─────────────────────────────────────────────
    content = []
    lang_name = translation.language if translation else "English"

    if has_image:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime};base64,{image_b64}", "detail": "high"},
        })
        instruction = (
            f"Analyze the banking customer query shown in this screenshot.\n"
            f"Write the summary and suggested_response_opening in {lang_name}."
        )
        if english_query:
            instruction += f"\n\nAdditional context (translated to English): {english_query}"
        instruction += "\n\nReturn the JSON response."
        content.append({"type": "text", "text": instruction})
    else:
        content.append({
            "type": "text",
            "text": (
                f"Customer query (translated to English): {english_query}\n"
                f"Write the summary and suggested_response_opening in {lang_name}.\n"
                f"Return the JSON response."
            ),
        })

    # ── Step 4b: LLM call ────────────────────────────────────────────────────
    print(f"🤖 Calling {model['label']}...")
    response = client.chat.completions.create(
        model=model["deployment"],
        messages=[
            {"role": "system", "content": build_system_prompt(rag_context)},
            {"role": "user",   "content": content},
        ],
        max_tokens=2048,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)

    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            analysis = json.loads(match.group())
        else:
            raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {raw[:300]}")

    # Early return for out-of-scope
    if analysis.get("out_of_scope"):
        return {
            "out_of_scope":    True,
            "message":         analysis.get("message"),
            "model_used":      model["label"],
            "detected_language": lang_name,
        }

    # ── Step 5: Translate output → English for CSR ────────────────────────────
    csr_translation = None
    if translation and not translation.is_english:
        print(f"🔄 Translating output from {translation.language} → English for CSR...")
        csr_translation = translate_output_to_english(
            analysis         = analysis,
            client           = client,
            deployment       = TEXT_MODEL["deployment"],
            source_language  = translation.language,
        )

    # ── Step 6: Assemble final response ──────────────────────────────────────
    result = {
        "out_of_scope": False,
        "analysis":     analysis,
        "model_used":   model["label"],
        # Translation metadata
        "translation": {
            "detected_language":  translation.language      if translation else "English",
            "language_code":      translation.language_code if translation else "en",
            "is_english":         translation.is_english    if translation else True,
            "original_query":     query_text or "",
            "english_query":      english_query,
            # CSR-facing English fields (populated when non-English detected)
            "summary_english":    csr_translation.summary_english   if csr_translation else analysis.get("summary", ""),
            "opening_english":    csr_translation.opening_english   if csr_translation else analysis.get("suggested_response_opening", ""),
            "translation_note":   csr_translation.translation_note  if csr_translation else "",
        },
        # RAG metadata for transparency
        "rag": {
            "context_length": len(rag_context),
            "query_used":     english_query[:120] + "..." if len(english_query) > 120 else english_query,
        },
    }
    return result


# ─── PDF generation ───────────────────────────────────────────────────────────
URG_COLORS = {"critical": HexColor("#C62828"), "high": HexColor("#E64A19"), "medium": HexColor("#F57C00"), "low": HexColor("#2E7D32")}
PRI_COLORS = {"immediate": HexColor("#C62828"), "standard": HexColor("#E64A19"), "low": HexColor("#2E7D32")}
NAVY  = HexColor("#1A2B5F"); GOLD = HexColor("#C9A84C"); ACCENT = HexColor("#1565C0")
TXT   = HexColor("#1A1A2E"); MUTED = HexColor("#6B7280"); CYAN = HexColor("#0277BD")
GREEN = HexColor("#2E7D32"); RED = HexColor("#C62828"); CARD = HexColor("#F8F9FC")


def generate_pdf(result: dict) -> str:
    analysis    = result["analysis"]
    model_label = result["model_used"]
    trans       = result.get("translation", {})

    filename = f"query_summary_{uuid.uuid4().hex[:8]}.pdf"
    doc = SimpleDocTemplate(
        str(PDF_DIR / filename),
        pagesize=A4,
        leftMargin=0.65*inch, rightMargin=0.65*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch,
    )

    ts2 = ParagraphStyle("t2", fontSize=17, leading=22, textColor=HexColor("#FFFFFF"), fontName=_F(bold=True))
    ss  = ParagraphStyle("s",  fontSize=11, leading=16, textColor=NAVY,  fontName=_F(bold=True), spaceBefore=12, spaceAfter=4)
    bs  = ParagraphStyle("b",  fontSize=10, leading=15, textColor=TXT,   fontName=_F(), spaceAfter=3)
    sm  = ParagraphStyle("sm", fontSize=9,  leading=13, textColor=MUTED, fontName=_F(italic=True))
    ml  = ParagraphStyle("ml", fontSize=7,  textColor=MUTED, fontName=_F(bold=True))
    fs  = ParagraphStyle("f",  fontSize=8,  textColor=MUTED, fontName=_F(), alignment=1)

    urg       = analysis.get("urgency", "medium").lower()
    urg_color = URG_COLORS.get(urg, ACCENT)
    story     = []

    def hr(color=HexColor("#E5E7EB")):
        return HRFlowable(width="100%", thickness=1, color=color, spaceAfter=5)

    def infocard(text, lc=ACCENT, style=None):
        t = Table([[Paragraph(text, style or bs)]], colWidths=[doc.width])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), CARD),
            ("LEFTPADDING",(0,0),(-1,-1),14), ("RIGHTPADDING",(0,0),(-1,-1),14),
            ("TOPPADDING", (0,0),(-1,-1),10), ("BOTTOMPADDING",(0,0),(-1,-1),10),
            ("LINEBEFORE", (0,0),(0,-1), 3, lc),
        ]))
        return t

    # Header
    title = analysis.get("query_title", "Query Summary")
    hdr = Table([[Paragraph(f"<b>{title}</b>", ts2)]], colWidths=[doc.width])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),NAVY),
        ("TOPPADDING",(0,0),(-1,-1),16), ("BOTTOMPADDING",(0,0),(-1,-1),16),
        ("LEFTPADDING",(0,0),(-1,-1),16), ("RIGHTPADDING",(0,0),(-1,-1),16),
        ("LINEABOVE",(0,0),(-1,0),4,GOLD),
    ]))
    story += [hdr, Spacer(1,8)]

    # Meta row
    escalation = "YES" if analysis.get("escalation_required") else "No"
    lang_display = trans.get("detected_language", "English")
    meta_cells = []
    for lbl, val in [["TYPE",analysis.get("query_type","General")], ["URGENCY",urg.upper()], ["LANGUAGE",lang_display], ["ESCALATION",escalation]]:
        vc = urg_color if lbl=="URGENCY" else (RED if val=="YES" else TXT)
        meta_cells.append([Paragraph(lbl,ml), Paragraph(f"<b>{val}</b>", ParagraphStyle("mvc",fontSize=9,textColor=vc,fontName=_F(bold=True)))])

    mt = Table([meta_cells], colWidths=[doc.width/4]*4)
    mt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),HexColor("#EEF2FF")),
        ("TOPPADDING",(0,0),(-1,-1),10), ("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1),12), ("RIGHTPADDING",(0,0),(-1,-1),12),
        ("LINEBEFORE",(1,0),(-1,-1),1,HexColor("#D1D5DB")),
    ]))
    story += [mt, Spacer(1,10)]

    # Summary (original language)
    story += [Paragraph("SUMMARY", ss), hr(), infocard(analysis.get("summary",""))]

    # English translation of summary (shown if non-English)
    if not trans.get("is_english", True) and trans.get("summary_english"):
        note_style = ParagraphStyle("note", fontSize=9, leading=14, textColor=CYAN, fontName=_F(italic=True))
        story += [Spacer(1,4), infocard(f"[CSR Translation] {trans['summary_english']}", lc=CYAN, style=note_style)]

    # Suggested opening (original language)
    if analysis.get("suggested_response_opening"):
        its = ParagraphStyle("its", fontSize=10, leading=15, textColor=HexColor("#1565C0"), fontName=_F(italic=True))
        story += [Spacer(1,6), Paragraph("SUGGESTED RESPONSE OPENING", ss), hr(),
                  infocard(f'"{analysis["suggested_response_opening"]}"', lc=CYAN, style=its)]

    # English translation of opening
    if not trans.get("is_english", True) and trans.get("opening_english"):
        note_style = ParagraphStyle("note2", fontSize=9, leading=14, textColor=MUTED, fontName=_F(italic=True))
        story += [Spacer(1,3), infocard(f'[CSR Translation] "{trans["opening_english"]}"', lc=MUTED, style=note_style)]

    # Key issues
    issues = analysis.get("key_issues",[])
    if issues:
        story += [Paragraph("KEY ISSUES", ss), hr()]
        for iss in issues:
            row = Table([[
                Paragraph(f"<b>{iss.get('issue','')}</b>", ParagraphStyle("ih",fontSize=10,textColor=ACCENT,fontName=_F(bold=True))),
                Paragraph(iss.get("detail",""), bs),
            ]], colWidths=[doc.width*0.28, doc.width*0.72])
            row.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),CARD),
                ("TOPPADDING",(0,0),(-1,-1),7), ("BOTTOMPADDING",(0,0),(-1,-1),7),
                ("LEFTPADDING",(0,0),(-1,-1),10), ("RIGHTPADDING",(0,0),(-1,-1),10),
                ("LINEBEFORE",(1,0),(1,-1),1,HexColor("#D1D5DB")),
            ]))
            story += [row, Spacer(1,3)]

    # Actions
    actions = analysis.get("recommended_actions",[])
    if actions:
        story += [Paragraph("RECOMMENDED CSR ACTIONS", ss), hr()]
        for i, act in enumerate(actions, 1):
            pri = act.get("priority","standard").lower()
            pc  = PRI_COLORS.get(pri, MUTED)
            ah = Table([[
                Paragraph(f"<b>#{i}  {act.get('action','')}</b>", ParagraphStyle("ah",fontSize=11,textColor=HexColor("#FFFFFF"),fontName=_F(bold=True))),
                Paragraph(f"<b>{pri.upper()}</b>", ParagraphStyle("pc",fontSize=8,textColor=pc,fontName=_F(bold=True),alignment=2)),
            ]], colWidths=[doc.width*0.78, doc.width*0.22])
            ah.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),NAVY),
                ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),8),
                ("LEFTPADDING",(0,0),(0,0),12), ("RIGHTPADDING",(-1,0),(-1,0),12),
                ("LINEABOVE",(0,0),(-1,0),2,pc),
            ]))
            ab = Table([[Paragraph(act.get("description",""), bs)]], colWidths=[doc.width])
            ab.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),CARD),
                ("LEFTPADDING",(0,0),(-1,-1),12), ("RIGHTPADDING",(0,0),(-1,-1),12),
                ("TOPPADDING",(0,0),(-1,-1),7), ("BOTTOMPADDING",(0,0),(-1,-1),7),
            ]))
            parts = [ah, ab]
            if act.get("team"):
                tt = Table([[Paragraph(f"Assigned to: <b>{act['team']}</b>", ParagraphStyle("tl",fontSize=8,textColor=MUTED,fontName=_F()))]], colWidths=[doc.width])
                tt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),CARD),("LEFTPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
                parts.append(tt)
            story += [KeepTogether(parts), Spacer(1,6)]

    if analysis.get("escalation_required") and analysis.get("escalation_reason"):
        story += [Paragraph("ESCALATION", ss), hr(RED), infocard(f"ALERT: {analysis['escalation_reason']}", lc=RED)]

    flags = analysis.get("compliance_flags",[])
    if flags:
        story += [Spacer(1,4), Paragraph("COMPLIANCE FLAGS", ss), hr(HexColor("#F59E0B")),
                  Paragraph("<br/>".join(f"* {f}" for f in flags), bs)]

    if analysis.get("estimated_resolution_time"):
        story += [Spacer(1,4), Paragraph("ESTIMATED RESOLUTION TIME", ss), hr(),
                  infocard(analysis["estimated_resolution_time"], lc=GREEN)]

    # Footer
    story.append(Spacer(1, 20))
    ft = Table([[Paragraph(
        f"Horizon Bank  |  BankQuery AI v3  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Model: {model_label}",
        fs,
    )]], colWidths=[doc.width])
    ft.setStyle(TableStyle([("LINEABOVE",(0,0),(-1,0),1,HexColor("#E5E7EB")),("TOPPADDING",(0,0),(-1,-1),6)]))
    story.append(ft)

    def draw_bg(canvas, _):
        canvas.saveState()
        canvas.setFillColor(HexColor("#FFFFFF"))
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_bg, onLaterPages=draw_bg)
    return filename


# ─── API Routes ───────────────────────────────────────────────────────────────


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return an inline SVG favicon — stops the 404 log noise."""
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
      <rect width="64" height="64" rx="12" fill="#1A2B5F"/>
      <polygon points="32,8 56,26 8,26" fill="#C9A84C"/>
      <rect x="8" y="28" width="48" height="5" rx="2" fill="#C9A84C"/>
      <rect x="12" y="34" width="6" height="18" rx="1" fill="#C9A84C"/>
      <rect x="22" y="34" width="6" height="18" rx="1" fill="#C9A84C"/>
      <rect x="36" y="34" width="6" height="18" rx="1" fill="#C9A84C"/>
      <rect x="46" y="34" width="6" height="18" rx="1" fill="#C9A84C"/>
      <rect x="8" y="52" width="48" height="4" rx="1" fill="#C9A84C"/>
    </svg>"""
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "version":      "3.0.0",
        "timestamp":    datetime.now().isoformat(),
        "vision_model": VISION_MODEL["label"],
        "text_model":   TEXT_MODEL["label"],
        "rag":          rag_stats(),
    }


@app.post("/summarize")
async def summarize(
    query_text:      Optional[str]        = Form(None),
    image:           Optional[UploadFile] = File(None),
    generate_report: bool                 = Form(True),
):
    image_b64 = image_mime = None

    if image and image.filename:
        raw_bytes  = await image.read()
        image_b64  = base64.b64encode(raw_bytes).decode()
        image_mime = image.content_type or "image/png"

    if not image_b64 and not query_text:
        raise HTTPException(status_code=400, detail="Please provide a screenshot or paste the query text.")

    result = summarize_query(query_text, image_b64, image_mime)

    if result.get("out_of_scope"):
        return JSONResponse(result)

    pdf_url = None
    if generate_report:
        pdf_url = f"/pdfs/{generate_pdf(result)}"

    result["pdf_url"] = pdf_url
    return JSONResponse(result)



@app.get("/languages")
def list_languages():
    """Return all supported languages for the UI dropdown."""
    return [{"code": k, "name": v} for k, v in SUPPORTED_LANGUAGES.items()]


@app.post("/translate")
async def translate_result(
    analysis_json:   str = Form(...),   # JSON string of the analysis dict
    target_language: str = Form(...),   # e.g. "Hindi", "French", "Tamil"
):
    """
    On-demand translation endpoint.
    Called when the CSR clicks the Translate button in the UI.
    Translates summary, opening, key issues, and actions into the chosen language.
    """
    try:
        analysis = json.loads(analysis_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid analysis JSON.")

    if not target_language.strip():
        raise HTTPException(status_code=400, detail="target_language is required.")

    client = get_client()
    result = translate_to_target_language(
        analysis        = analysis,
        target_language = target_language,
        client          = client,
        deployment      = TEXT_MODEL["deployment"],
    )

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Translation failed."))

    return JSONResponse(result)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)