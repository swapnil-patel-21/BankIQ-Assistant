"""
rag.py — Retrieval-Augmented Generation module for Horizon Bank Knowledge Base.

How it works:
  1. On startup, loads bank_knowledge.json and splits it into meaningful text chunks.
  2. Builds a TF-IDF matrix over all chunks (no external embedding API needed).
  3. At query time, vectorizes the incoming query and computes cosine similarity
     against all chunks, returning the top-K most relevant ones.
  4. The retrieved chunks are injected into the LLM prompt instead of the
     full knowledge base — keeping the prompt lean and focused.

Usage from main.py:
    from rag import retrieve

    context = retrieve("credit card interest calculation EMI")
    # context is a plain string ready to inject into the system prompt
"""

import json
import re
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────
KB_PATH   = Path(__file__).parent / "bank_knowledge.json"
TOP_K     = 5      # number of chunks to retrieve per query
MIN_SCORE = 0.05   # minimum similarity score to include a chunk


# ── Chunk builder ─────────────────────────────────────────────────────────────

def _flatten(obj, prefix: str = "") -> List[str]:
    """
    Recursively flatten a nested dict/list into readable text chunks.
    Each chunk is a self-contained paragraph with context from its key path.
    """
    chunks = []

    if isinstance(obj, dict):
        for key, val in obj.items():
            label = key.replace("_", " ").title()
            path  = f"{prefix} > {label}" if prefix else label

            if isinstance(val, (dict, list)):
                chunks.extend(_flatten(val, path))
            else:
                chunks.append(f"{path}: {val}")

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                # Try to make a readable sentence from dict items
                parts = []
                for k, v in item.items():
                    if not isinstance(v, (dict, list)):
                        parts.append(f"{k.replace('_',' ')}: {v}")
                if parts:
                    chunks.append(f"{prefix} — " + ", ".join(parts))
                chunks.extend(_flatten(item, prefix))
            elif isinstance(item, str):
                chunks.append(f"{prefix}: {item}")
            else:
                chunks.extend(_flatten(item, prefix))

    return chunks


def _build_chunks(kb: dict) -> List[str]:
    """
    Build domain-aware chunks from the bank knowledge base.
    Produces both fine-grained key-value chunks AND
    higher-level section summaries for better retrieval coverage.
    """
    chunks = []

    # ── FAQs — most important, keep intact as full Q&A pairs ──
    for faq in kb.get("faqs", []):
        q = faq.get("q", "")
        a = faq.get("a", "")
        chunks.append(f"FAQ: {q}\nAnswer: {a}")

    # ── Credit cards — critical for billing queries ──
    cc = kb.get("credit_cards", {})
    for card_key, card in cc.items():
        if not isinstance(card, dict):
            continue
        name = card_key.replace("_", " ").title()

        # Card overview chunk
        overview_parts = []
        for field in ["annual_fee", "interest_rate_monthly", "minimum_due",
                      "credit_limit", "reward_points", "interest_free_period"]:
            if field in card:
                overview_parts.append(f"{field.replace('_',' ')}: {card[field]}")
        if overview_parts:
            chunks.append(f"Credit Card — {name} overview:\n" + "\n".join(overview_parts))

        # EMI conversion chunk (kept separate for targeted retrieval)
        emi = card.get("emi_conversion", {})
        if emi and emi.get("available"):
            rates = emi.get("interest_rates", {})
            rate_lines = "\n".join(f"  {k.replace('_',' ')}: {v}" for k, v in rates.items())
            chunks.append(
                f"Credit Card — {name} EMI Conversion:\n"
                f"Minimum amount: {emi.get('minimum_amount','')}\n"
                f"Available tenures (months): {', '.join(str(t) for t in emi.get('tenures',[]))}\n"
                f"Interest rates per tenure:\n{rate_lines}\n"
                f"Processing fee: {emi.get('processing_fee','')}\n"
                f"Foreclosure charge: {emi.get('foreclosure_charge','')}"
            )

        # Fees chunk
        fee_fields = ["late_payment_fee", "cash_advance_fee", "foreign_transaction_fee", "cheque_bounce"]
        fee_parts  = [f"{f.replace('_',' ')}: {card[f]}" for f in fee_fields if f in card]
        if fee_parts:
            chunks.append(f"Credit Card — {name} fees & charges:\n" + "\n".join(fee_parts))

    # Billing policy (interest-free period, billing cycle) as a separate chunk
    for field in ["interest_free_period", "billing_cycle", "billing_note"]:
        if field in cc:
            chunks.append(f"Credit Card Billing Policy — {field.replace('_',' ')}: {cc[field]}")

    # ── Fixed deposits ──
    fd = kb.get("fixed_deposits", {})
    rates_text = "\n".join(
        f"  {r['tenure']}: {r['rate']}"
        for r in fd.get("general_rates", [])
    )
    if rates_text:
        chunks.append(
            f"Fixed Deposit Interest Rates:\n{rates_text}\n"
            f"Senior citizen additional: {fd.get('senior_citizen_additional','')}\n"
            f"Premature withdrawal penalty: {fd.get('premature_withdrawal_penalty','')}\n"
            f"Minimum amount: {fd.get('minimum_amount','')}"
        )
    tfd = fd.get("tax_saving_fd", {})
    if tfd:
        chunks.append(
            f"Tax Saving FD (Section 80C):\n"
            f"Tenure: {tfd.get('tenure','')}\n"
            f"Rate: {tfd.get('rate','')}\n"
            f"Max deduction: {tfd.get('max_deduction_80C','')}"
        )

    # ── Savings accounts ──
    for acc_key, acc in kb.get("savings_accounts", {}).items():
        if isinstance(acc, dict):
            name = acc_key.replace("_", " ").title()
            parts = [f"{k.replace('_',' ')}: {v}" for k, v in acc.items() if not isinstance(v, list)]
            perks = acc.get("perks", [])
            if perks:
                parts.append("Perks: " + ", ".join(perks))
            chunks.append(f"Savings Account — {name}:\n" + "\n".join(parts))

    # ── Loans ──
    for loan_key, loan in kb.get("loans", {}).items():
        if isinstance(loan, dict):
            name = loan_key.replace("_", " ").title()
            parts = [f"{k.replace('_',' ')}: {v}" for k, v in loan.items() if not isinstance(v, list)]
            chunks.append(f"Loan — {name}:\n" + "\n".join(parts))

    # ── Charges and fees ──
    charges = kb.get("charges_and_fees", {})
    flat_charges = _flatten(charges, "Charges and Fees")
    if flat_charges:
        chunks.append("Banking Charges & Fees:\n" + "\n".join(flat_charges))

    # ── Digital banking ──
    digital = kb.get("digital_banking", {})
    flat_digital = _flatten(digital, "Digital Banking")
    if flat_digital:
        chunks.append("Digital Banking Features:\n" + "\n".join(flat_digital))

    # ── KYC documents ──
    kyc = kb.get("kyc_documents", {})
    flat_kyc = _flatten(kyc, "KYC Documents")
    if flat_kyc:
        chunks.append("KYC & Documentation Requirements:\n" + "\n".join(flat_kyc))

    # ── Dispute & grievance ──
    dg = kb.get("dispute_and_grievance", {})
    flat_dg = _flatten(dg, "Dispute & Grievance")
    if flat_dg:
        chunks.append("Dispute Resolution & Grievance Process:\n" + "\n".join(flat_dg))

    # ── Bank contact info ──
    chunks.append(
        f"Bank Contact Information:\n"
        f"Name: {kb.get('bank_name','Horizon Bank')}\n"
        f"Customer care: {kb.get('customer_care','')}\n"
        f"Email: {kb.get('email','')}\n"
        f"Headquarters: {kb.get('headquarters','')}"
    )

    # Filter out empty chunks
    return [c.strip() for c in chunks if c.strip()]


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        self.chunks: List[str] = []
        self.vectorizer: TfidfVectorizer = None
        self.matrix = None
        self._load()

    def _load(self):
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                kb = json.load(f)

            self.chunks = _build_chunks(kb)

            # TF-IDF with bigrams for better phrase matching
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_df=1000000,
                sublinear_tf=True,       # log(1+tf) — dampens high-frequency terms
                strip_accents="unicode",
            )
            self.matrix = self.vectorizer.fit_transform(self.chunks)

            print(f"✅ RAG: Loaded {len(self.chunks)} knowledge chunks from {KB_PATH.name}")

        except FileNotFoundError:
            print(f"⚠  RAG: {KB_PATH} not found. RAG will return empty context.")
        except Exception as e:
            print(f"⚠  RAG: Failed to load knowledge base: {e}")

    def retrieve(self, query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> str:
        """
        Retrieve the top-K most relevant knowledge chunks for the given query.

        Args:
            query:     The customer query (in any language — works best with
                       English or transliterated text; call after translation).
            top_k:     Number of chunks to return.
            min_score: Minimum cosine similarity to include a chunk.

        Returns:
            A formatted string of retrieved chunks, ready to inject into a prompt.
        """
        if not self.chunks or self.vectorizer is None:
            return "No knowledge base available."

        query_vec = self.vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, self.matrix).flatten()

        # Sort by score descending, take top_k above threshold
        top_indices = np.argsort(scores)[::-1]
        selected = [
            (i, scores[i]) for i in top_indices
            if scores[i] >= min_score
        ][:top_k]

        if not selected:
            # Fallback: always return at least the FAQs
            faq_chunks = [c for c in self.chunks if c.startswith("FAQ:")][:3]
            if faq_chunks:
                return "Relevant Knowledge (fallback — FAQs):\n\n" + "\n\n---\n\n".join(faq_chunks)
            return "No highly relevant knowledge found for this query."

        retrieved = []
        for rank, (idx, score) in enumerate(selected, 1):
            retrieved.append(
                f"[Chunk {rank} | Relevance: {score:.2f}]\n{self.chunks[idx]}"
            )

        return "Relevant Knowledge Base Excerpts:\n\n" + "\n\n---\n\n".join(retrieved)

    def get_stats(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
            "vectorizer_ready": self.vectorizer is not None,
        }


# ── Singleton instance (imported by main.py) ─────────────────────────────────
_engine = RAGEngine()


def retrieve(query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> str:
    """
    Public API. Call this from main.py:
        from rag import retrieve
        context = retrieve("credit card interest EMI conversion")
    """
    return _engine.retrieve(query, top_k, min_score)


def get_stats() -> dict:
    """Return RAG engine statistics for health endpoint."""
    return _engine.get_stats()


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "credit card interest minimum due EMI conversion",
        "unauthorized transaction fraud dispute RBI",
        "home loan interest rate processing fee",
        "KYC name change marriage certificate",
        "fixed deposit interest rate senior citizen",
        "net banking locked password reset",
    ]
    print("\n" + "="*60)
    print("RAG RETRIEVAL TEST")
    print("="*60)
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 40)
        result = retrieve(q, top_k=2)
        print(result[:500] + ("..." if len(result) > 500 else ""))
        print()
