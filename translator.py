"""
translator.py — Language detection, input translation, and on-demand output translation
                for Horizon Bank CSR Tool.

Functions exported (called from main.py):
  - detect_and_translate_input(text, client, deployment)
      → TranslatedInput: detects language, translates query to English for RAG + LLM

  - translate_output_to_english(analysis, client, deployment, source_language)
      → TranslatedOutput: translates AI summary/opening to English for CSR auto-display

  - translate_to_target_language(texts, target_language, client, deployment)
      → dict: on-demand translation of any text fields to any language chosen by CSR
         Called by the /translate API endpoint when user clicks the translate button
"""

import json
import re
from dataclasses import dataclass
from typing import Optional


# ── Supported languages (shown in the UI dropdown) ───────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TranslatedInput:
    original_text:  str
    english_text:   str
    language:       str
    language_code:  str
    is_english:     bool
    confidence:     str


@dataclass
class TranslatedOutput:
    summary_english:   str
    opening_english:   str
    translation_note:  str


# ── 1. Detect language + translate input → English ────────────────────────────

DETECT_PROMPT = """You are a precise language detection and translation assistant.

Given the input text:
1. Detect its language.
2. If NOT English, translate it to clear natural English.
3. If already English, return it unchanged.

Return ONLY this JSON (no markdown):
{
  "language": "Full language name in English e.g. Hindi, Tamil",
  "language_code": "ISO 639-1 code e.g. hi, ta, en",
  "is_english": true or false,
  "confidence": "high | medium | low",
  "english_text": "English translation or original if already English"
}"""


def detect_and_translate_input(text: str, client, deployment: str) -> TranslatedInput:
    if not text or not text.strip():
        return TranslatedInput("", "", "English", "en", True, "high")

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": DETECT_PROMPT},
                {"role": "user",   "content": f"Text:\n\n{text}"},
            ],
            max_tokens=600, temperature=0.0,
        )
        raw = _clean_json(resp.choices[0].message.content)
        p   = json.loads(raw)
        return TranslatedInput(
            original_text = text,
            english_text  = p.get("english_text", text),
            language      = p.get("language", "Unknown"),
            language_code = p.get("language_code", "??"),
            is_english    = p.get("is_english", True),
            confidence    = p.get("confidence", "medium"),
        )
    except Exception as e:
        print(f"⚠ detect_and_translate_input failed: {e}")
        return TranslatedInput(text, text, "Unknown", "??", True, "low")


# ── 2. Translate AI output → English (auto, for CSR display) ─────────────────

AUTO_EN_PROMPT = """You are a professional banking translation assistant.
Translate the provided text to clear professional English.
Keep all monetary amounts, figures, dates, and technical terms exactly as-is.

Return ONLY this JSON:
{
  "summary_english": "...",
  "opening_english": "..."
}"""


def translate_output_to_english(
    analysis: dict, client, deployment: str, source_language: str = "Unknown"
) -> TranslatedOutput:
    summary = analysis.get("summary", "")
    opening = analysis.get("suggested_response_opening", "")

    if not summary and not opening:
        return TranslatedOutput("", "", "")

    try:
        payload = json.dumps({"summary": summary, "suggested_response_opening": opening}, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": AUTO_EN_PROMPT},
                {"role": "user",   "content": f"Translate to English:\n\n{payload}"},
            ],
            max_tokens=600, temperature=0.0,
        )
        raw = _clean_json(resp.choices[0].message.content)
        p   = json.loads(raw)
        return TranslatedOutput(
            summary_english  = p.get("summary_english", summary),
            opening_english  = p.get("opening_english", opening),
            translation_note = f"Auto-translated from {source_language} for CSR reference",
        )
    except Exception as e:
        print(f"⚠ translate_output_to_english failed: {e}")
        return TranslatedOutput(summary, opening, f"Translation from {source_language} failed")


# ── 3. On-demand translate to ANY target language (CSR-chosen) ────────────────

ON_DEMAND_PROMPT = """You are a professional banking translation assistant.
Translate ALL provided text fields into {target_language}.
Keep monetary amounts (₹, $), figures, proper nouns, and technical banking terms exactly as-is.
The translation must sound natural and professional in {target_language}.

Return ONLY this JSON (no markdown, no explanation):
{{
  "summary": "translated summary",
  "suggested_response_opening": "translated opening",
  "key_issues": [
    {{"issue": "translated label", "detail": "translated detail"}}
  ],
  "recommended_actions": [
    {{"action": "translated action title", "description": "translated description"}}
  ]
}}"""


def translate_to_target_language(
    analysis: dict,
    target_language: str,
    client,
    deployment: str,
) -> dict:
    """
    On-demand: translate the full analysis output into any CSR-chosen language.
    Called by the POST /translate endpoint in main.py.

    Args:
        analysis:        Full analysis dict from the LLM.
        target_language: Language name chosen by CSR e.g. "French", "Hindi", "Tamil".
        client:          OpenAI client.
        deployment:      Model deployment name.

    Returns:
        dict with translated fields ready to merge into the UI.
    """
    # Build a compact payload with only the text fields that need translation
    payload = {
        "summary":                      analysis.get("summary", ""),
        "suggested_response_opening":   analysis.get("suggested_response_opening", ""),
        "key_issues": [
            {"issue": i.get("issue", ""), "detail": i.get("detail", "")}
            for i in analysis.get("key_issues", [])
        ],
        "recommended_actions": [
            {"action": a.get("action", ""), "description": a.get("description", "")}
            for a in analysis.get("recommended_actions", [])
        ],
    }

    system = ON_DEMAND_PROMPT.format(target_language=target_language)

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Translate to {target_language}:\n\n{json.dumps(payload, ensure_ascii=False)}"},
            ],
            max_tokens=1500, temperature=0.1,
        )
        raw = _clean_json(resp.choices[0].message.content)
        translated = json.loads(raw)
        translated["target_language"] = target_language
        translated["success"]         = True
        return translated

    except Exception as e:
        print(f"⚠ translate_to_target_language failed: {e}")
        return {
            "success":         False,
            "target_language": target_language,
            "error":           str(e),
        }


# ── Helper ────────────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    return text


def language_display(code: str, name: str) -> str:
    return f"{name} ({code})" if code != "??" else name
