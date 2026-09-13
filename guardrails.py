"""
PolicyCheck AI — Guardrails Module
Provides Input, Scope, Safety, Retrieval Relevance, and Output Groundedness Guardrails.
"""

import re
import os
import math
import requests
from typing import Dict, Any, List, Tuple

# Threshold for similarity score to determine if knowledge base has sufficient info
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.50"))

# Keywords related to the application's intended domain (Indian Tax, Policy, Law, Regulations)
IN_SCOPE_KEYWORDS = {
    "gst", "tax", "income tax", "cgst", "sgst", "igst", "utgst", "slab", "rate", "exempt",
    "exemption", "return", "filing", "audit", "compliance", "policy", "act", "section",
    "invoice", "credit", "itc", "input tax credit", "penalty", "interest", "refund",
    "assessment", "tds", "tcs", "deduction", "assessment", "company", "director", "board",
    "share", "pf", "epf", "gratuity", "esi", "labour", "wages", "minimum wage", "leave",
    "maternity", "factories", "bonus", "fdi", "export", "import", "sez", "rbi", "sebi",
    "corporate", "governance", "notice", "appeal", "tribunal", "rule", "circular", "notification"
}

# Distinct patterns indicating off-topic / out-of-scope requests
OUT_OF_SCOPE_PATTERNS = [
    r"\b(write|create|code|generate)\b.*\b(python|javascript|java|c\+\+|html|css|script|program|app|game)\b",
    r"\b(recipe|cook|bake|ingredients|dish|food)\b",
    r"\b(capital of|population of|distance to|who won|movie|film|song|actor|actress|sports|cricket|football)\b",
    r"\b(write a (poem|story|essay|song|joke|rap|script))\b",
    r"\b(translate this to|french|spanish|german|chinese)\b",
    r"\b(weather in|forecast|temperature in)\b",
    r"\b(diagnose|symptoms of|treatment for|cure for|medicine)\b",
    r"\b(solve|calculate)\b.*\b(math|equation|integral|derivative)\b"
]

# Patterns indicating prompt injection, system prompt leak, or safety violations
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"you\s+are\s+now\s+a",
    r"system\s+prompt",
    r"reveal\s+your\s+instructions",
    r"disregard\s+(your\s+)?(system\s+)?prompt",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"jailbreak"
]


class InputGuardrail:
    """Evaluates input prompts for length, scope, safety, and formatting."""

    @staticmethod
    def check_length(query: str, max_chars: int = 1000, max_words: int = 200) -> Tuple[bool, str]:
        if not query or not query.strip():
            return False, "INPUT_EMPTY: Query is empty."
        
        char_count = len(query)
        word_count = len(query.split())
        
        if char_count > max_chars:
            return False, f"INPUT_TOO_LONG: Query length ({char_count} chars) exceeds maximum allowed limit of {max_chars} characters."
        
        if word_count > max_words:
            return False, f"INPUT_TOO_LONG: Query word count ({word_count} words) exceeds maximum allowed limit of {max_words} words."
        
        return True, "PASSED"

    @staticmethod
    def check_scope(query: str) -> Tuple[bool, str]:
        q_lower = query.lower()

        # Check explicit off-topic patterns
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, q_lower):
                return False, "OUT_OF_SCOPE: Query requests content or actions outside PolicyCheck AI's policy/tax domain (e.g. coding, creative writing, trivia)."

        # Check domain relevance (if query has > 4 words, expect at least some policy relevance or general policy question wording)
        words = set(re.findall(r"[a-z]{3,}", q_lower))
        stop_words = {"what", "is", "the", "for", "how", "can", "are", "under", "which", "when", "who", "does", "policy", "check"}
        content_words = words - stop_words
        
        # If it's a specific question and contains zero policy domain keywords while matching general non-policy subjects
        if len(content_words) >= 3 and not any(kw in q_lower for kw in IN_SCOPE_KEYWORDS):
            # Check if query is clearly general non-policy topic
            non_policy_topics = ["weather", "recipe", "python", "code", "movie", "football", "planet", "solar system", "physics", "chemistry", "biology", "history ofrome"]
            if any(tp in q_lower for tp in non_policy_topics):
                return False, "OUT_OF_SCOPE: Topic is unrelated to corporate, tax, or regulatory policy."

        return True, "PASSED"

    @staticmethod
    def check_safety_and_injection(query: str) -> Tuple[bool, str]:
        q_lower = query.lower()

        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, q_lower):
                return False, "PROMPT_INJECTION: Potential prompt injection or system prompt override attempt detected."

        return True, "PASSED"


class RetrievalGuardrail:
    """Evaluates retrieval quality to determine if sufficient information exists in the knowledge base."""

    @staticmethod
    def check_information_availability(results: List[Dict[str, Any]], threshold: float = RELEVANCE_THRESHOLD) -> Tuple[bool, str, float]:
        if not results:
            return False, "INSUFFICIENT_INFORMATION: No matching documents retrieved from knowledge base.", 0.0

        best_score = results[0].get("score", 0.0)
        
        if best_score < threshold:
            return (
                False,
                f"INSUFFICIENT_INFORMATION: Highest retrieval similarity score ({best_score:.4f}) is below relevance threshold ({threshold:.2f}). Query cannot be reliably answered from knowledge base.",
                best_score
            )

        return True, "PASSED", best_score


class OutputGuardrail:
    """Evaluates generated LLM outputs for groundedness, factual consistency, unsupported claims, and format."""

    @staticmethod
    def check_groundedness_and_claims(answer: str, context: str) -> Tuple[bool, str, List[str]]:
        """
        Scans answer for specific claims (e.g. tax rates, percentages, section numbers, amounts)
        and verifies if they exist in the retrieved context.
        """
        if not answer or not answer.strip():
            return False, "OUTPUT_EMPTY: Model returned empty response.", []

        # Refusal answers are inherently grounded
        refusal_phrases = ["could not find", "not available", "no information", "outside", "cannot find", "not present"]
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            return True, "PASSED: Controlled refusal response.", []

        unsupported_claims = []
        ctx_lower = context.lower()

        # 1. Extract percentages (e.g., 18%, 28%, 5%)
        percentages = re.findall(r"\b\d+(?:\.\d+)?\s*%\b", answer)
        for pct in percentages:
            normalized_pct = pct.replace(" ", "")
            if normalized_pct.lower() not in ctx_lower.replace(" ", ""):
                unsupported_claims.append(f"Percentage rate '{pct}' not found in retrieved context.")

        # 2. Extract section numbers (e.g., Section 16, Section 54(3))
        sections = re.findall(r"\bSection\s+\d+(?:\(\d+\))?\b", answer, re.IGNORECASE)
        for sec in sections:
            if sec.lower() not in ctx_lower:
                unsupported_claims.append(f"Statutory reference '{sec}' not mentioned in retrieved context.")

        # 3. Extract monetary figures (e.g., 10 lakh, 50,000, Rs. 20,000)
        amounts = re.findall(r"\b(?:rs\.?|inr)?\s*\d{1,3}(?:,\d{2,3})*(?:\s*lakh|\s*crore)?\b", answer, re.IGNORECASE)
        for amt in amounts:
            amt_clean = amt.strip().lower()
            if len(amt_clean) > 2 and amt_clean not in ctx_lower:
                # Exclude trivial single digit numbers
                if not re.match(r"^\d$", amt_clean):
                    unsupported_claims.append(f"Monetary figure '{amt}' not present in context.")

        if unsupported_claims:
            return False, f"UNSUPPORTED_CLAIMS: Generated answer contains {len(unsupported_claims)} claims/figures unsupported by context.", unsupported_claims

        return True, "PASSED", []

    @staticmethod
    def check_output_format(answer: str, max_length: int = 4000) -> Tuple[bool, str]:
        if not answer or not answer.strip():
            return False, "FORMAT_VIOLATION: Empty output generated."
        
        if len(answer) > max_length:
            return False, f"FORMAT_VIOLATION: Output length ({len(answer)} chars) exceeds maximum permitted size of {max_length}."
        
        # Check for system prompt leaks
        system_leak_triggers = ["You are PolicyCheck AI", "Answer the user's question using ONLY", "Context:"]
        for trigger in system_leak_triggers:
            if answer.count(trigger) > 1 or (trigger in answer and len(answer) < 150):
                return False, f"SYSTEM_PROMPT_LEAK: Response includes raw system prompt instructions ('{trigger}')."

        return True, "PASSED"


class GuardrailManager:
    """Unified Orchestrator for running all Input, Retrieval, and Output Guardrails."""

    def __init__(self, relevance_threshold: float = RELEVANCE_THRESHOLD):
        self.relevance_threshold = relevance_threshold

    def validate_input(self, query: str) -> Dict[str, Any]:
        """Runs input length, safety, and scope guardrails."""
        # 1. Length Check
        passed, reason = InputGuardrail.check_length(query)
        if not passed:
            return {"passed": False, "guardrail": "INPUT_LENGTH", "reason": reason}

        # 2. Safety & Injection Check
        passed, reason = InputGuardrail.check_safety_and_injection(query)
        if not passed:
            return {"passed": False, "guardrail": "SAFETY_INJECTION", "reason": reason}

        # 3. Scope Check
        passed, reason = InputGuardrail.check_scope(query)
        if not passed:
            return {"passed": False, "guardrail": "SCOPE_FILTER", "reason": reason}

        return {"passed": True, "guardrail": "INPUT_ALL", "reason": "PASSED"}

    def validate_retrieval(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs retrieval relevance guardrail."""
        passed, reason, score = RetrievalGuardrail.check_information_availability(results, self.relevance_threshold)
        return {
            "passed": passed,
            "guardrail": "RETRIEVAL_RELEVANCE",
            "reason": reason,
            "best_score": score,
            "threshold": self.relevance_threshold
        }

    def validate_output(self, answer: str, context: str) -> Dict[str, Any]:
        """Runs output format and groundedness guardrails."""
        # 1. Format Check
        passed, reason = OutputGuardrail.check_output_format(answer)
        if not passed:
            return {"passed": False, "guardrail": "OUTPUT_FORMAT", "reason": reason, "unsupported_claims": []}

        # 2. Groundedness & Claim Verification
        passed, reason, claims = OutputGuardrail.check_groundedness_and_claims(answer, context)
        if not passed:
            return {"passed": False, "guardrail": "GROUNDEDNESS_CLAIMS", "reason": reason, "unsupported_claims": claims}

        return {"passed": True, "guardrail": "OUTPUT_ALL", "reason": "PASSED", "unsupported_claims": []}
