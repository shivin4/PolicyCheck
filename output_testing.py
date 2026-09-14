"""
PolicyCheck AI — AI Output Testing Framework
Treats LLM output as data that must be systematically tested against defined conditions
before acceptance by the application.
"""

import re
from typing import Dict, Any, List, Tuple


def evaluate_relevance(question: str, answer: str) -> Tuple[bool, float, str]:
    """
    Condition 1: Is the answer relevant to the question?
    Evaluates topical relevance by calculating key question keyword coverage in answer.
    """
    if not answer or not answer.strip():
        return False, 0.0, "Answer is empty."

    stop = {"what", "is", "the", "for", "how", "can", "are", "under", "which", "when", "who", "does", "policy", "check", "with", "from", "have", "been"}
    q_words = set(w.lower() for w in re.findall(r"[a-z]{3,}", question) if w.lower() not in stop)
    
    if not q_words:
        return True, 1.0, "Question contains only general stopwords."

    ans_lower = answer.lower()
    matched = set(w for w in q_words if w in ans_lower)
    score = round(len(matched) / len(q_words), 4)

    if any(phrase in ans_lower for phrase in ["not found", "could not find", "not available", "no information"]):
        return True, 1.0, "Response appropriately addresses question by declaring information unavailable."

    passed = score >= 0.25
    reason = f"Keyword relevance score: {score:.2f} ({len(matched)}/{len(q_words)} question terms matched)." if passed else f"Low relevance score: {score:.2f}. Answer misses core question topics."
    return passed, score, reason


def evaluate_groundedness(context: str, answer: str) -> Tuple[bool, float, str]:
    """
    Condition 2: Is the answer supported by the retrieved context?
    Checks the proportion of key informational terms in the answer that are present in context.
    """
    if not answer or not answer.strip():
        return False, 0.0, "Answer is empty."

    ans_lower = answer.lower()
    if any(phrase in ans_lower for phrase in ["not found", "could not find", "not available", "no information", "outside"]):
        return True, 1.0, "Response is a controlled refusal statement."

    if not context or not context.strip():
        return False, 0.0, "Context is empty while answer contains factual assertions."

    ctx_lower = context.lower()
    ans_words = [w for w in re.findall(r"[a-z]{4,}", ans_lower) if w not in {"this", "that", "from", "with", "have", "been", "they", "will", "which", "where", "shall"}]

    if not ans_words:
        return True, 1.0, "No complex informational terms in answer."

    supported_count = sum(1 for w in ans_words if w in ctx_lower)
    groundedness_score = round(supported_count / len(ans_words), 4)

    passed = groundedness_score >= 0.60
    reason = f"Groundedness score: {groundedness_score:.2f} ({supported_count}/{len(ans_words)} terms supported by context)." if passed else f"Low groundedness: {groundedness_score:.2f}. Many answer terms absent from retrieved context."
    return passed, groundedness_score, reason


def evaluate_unsupported_claims(context: str, answer: str) -> Tuple[bool, List[str], str]:
    """
    Condition 3: Does it contain unsupported claims?
    Checks for explicit numbers, tax rates, section numbers, or statutory values not found in context.
    """
    if not answer or not answer.strip():
        return True, [], "No text to check for unsupported claims."

    ans_lower = answer.lower()
    if any(phrase in ans_lower for phrase in ["not found", "could not find", "not available"]):
        return True, [], "No claims made (refusal response)."

    ctx_lower = context.lower() if context else ""
    unsupported = []

    # 1. Check percentages (e.g. 18%, 28%)
    pcts = re.findall(r"\b\d+(?:\.\d+)?\s*%\b", answer)
    for pct in pcts:
        if pct.replace(" ", "").lower() not in ctx_lower.replace(" ", ""):
            unsupported.append(f"Percentage '{pct}'")

    # 2. Check statutory section numbers (e.g. Section 16, Section 66A)
    sections = re.findall(r"\bSection\s+(\d+[A-Z]?(?:\(\d+\))?)\b", answer, re.IGNORECASE)
    for sec_num in sections:
        sec_clean = sec_num.lower()
        if sec_clean not in ctx_lower and f"section {sec_clean}" not in ctx_lower and f"sec {sec_clean}" not in ctx_lower:
            unsupported.append(f"Statutory reference 'Section {sec_num}'")

    # 3. Check explicit monetary figures (e.g. Rs. 20,000, 10 lakh)
    amounts = re.findall(r"\b(?:rs\.?|inr)\s*\d+(?:,\d+)*(?:\s*lakh|\s*crore)?\b|\b\d{1,3}(?:,\d{3})+(?:\s*lakh|\s*crore)?\b", answer, re.IGNORECASE)
    for amt in amounts:
        amt_clean = amt.strip().lower()
        if len(amt_clean) > 2 and amt_clean not in ctx_lower:
            unsupported.append(f"Monetary figure '{amt}'")

    passed = len(unsupported) == 0
    reason = "No unsupported numerical or statutory claims detected." if passed else f"Detected {len(unsupported)} unsupported claims: {', '.join(unsupported)}."
    return passed, unsupported, reason


def evaluate_format_compliance(answer: str, max_chars: int = 4000) -> Tuple[bool, str]:
    """
    Condition 4: Does it follow the expected format?
    Checks non-emptiness, character length limits, clean markdown/text, and absence of system prompt leakage.
    """
    if not answer or not answer.strip():
        return False, "FAIL: Output is empty."

    if len(answer) > max_chars:
        return False, f"FAIL: Output exceeds max length ({len(answer)} > {max_chars} chars)."

    leak_triggers = ["You are PolicyCheck AI", "Answer the user's question using ONLY", "Context:"]
    for trigger in leak_triggers:
        if trigger in answer and len(answer) < 150:
            return False, f"FAIL: System prompt leak detected ('{trigger}')."

    return True, "PASS: Response complies with format, structure, and length boundaries."


def evaluate_answer_sufficiency(kb_supported: bool, context: str, answer: str) -> Tuple[bool, str]:
    """
    Condition 5: Does it provide an answer when sufficient information exists?
    When kb_supported=True and context is available, verifies that the model provided an actual answer
    instead of inappropriately refusing.
    """
    if not kb_supported:
        return True, "N/A: Question is not supported by KB."

    ans_lower = answer.lower()
    refusal_phrases = ["could not find this information", "information is not available", "cannot find", "no information"]

    if any(phrase in ans_lower for phrase in refusal_phrases) and len(answer) < 120:
        return False, "FAIL: Model inappropriately refused to answer despite sufficient context in KB."

    return True, "PASS: Model successfully generated answer from available context."


def evaluate_appropriate_refusal(kb_supported: bool, answer: str) -> Tuple[bool, str]:
    """
    Condition 6: Does it appropriately refuse when information is unavailable?
    When kb_supported=False or query is out of scope, verifies that the model declined to answer.
    """
    if kb_supported:
        return True, "N/A: Question is supported by KB."

    ans_lower = answer.lower()
    refusal_phrases = [
        "not found", "not available", "could not find", "no information",
        "outside", "cannot find", "not present", "not in the policy documents",
        "rejected", "out_of_scope", "insufficient_information"
    ]

    has_refusal = any(phrase in ans_lower for phrase in refusal_phrases)
    has_hallucinated_number = bool(re.search(r"\b(5|12|18|28)\s*%\b", ans_lower))

    if has_refusal and not has_hallucinated_number:
        return True, "PASS: Model appropriately refused to answer out-of-scope or unavailable question."
    elif has_hallucinated_number:
        return False, "FAIL: Model hallucinated specific numbers instead of refusing."
    else:
        return False, "FAIL: Model failed to provide a clear refusal for unavailable information."


def run_output_test_case(test_case: Dict[str, Any], answer: str, context: str) -> Dict[str, Any]:
    """
    Runs all 6 AI output tests on a generated response for a given test case.
    """
    question = test_case["question"]
    kb_supported = test_case.get("kb_supported", True)

    c1_pass, c1_score, c1_reason = evaluate_relevance(question, answer)
    c2_pass, c2_score, c2_reason = evaluate_groundedness(context, answer)
    c3_pass, c3_claims, c3_reason = evaluate_unsupported_claims(context, answer)
    c4_pass, c4_reason = evaluate_format_compliance(answer)
    c5_pass, c5_reason = evaluate_answer_sufficiency(kb_supported, context, answer)
    c6_pass, c6_reason = evaluate_appropriate_refusal(kb_supported, answer)

    all_passed = all([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])

    return {
        "test_id": test_case.get("id", "UNKNOWN"),
        "question": question,
        "kb_supported": kb_supported,
        "overall_status": "PASS" if all_passed else "FAIL",
        "conditions": {
            "relevance": {"passed": c1_pass, "score": c1_score, "reason": c1_reason},
            "groundedness": {"passed": c2_pass, "score": c2_score, "reason": c2_reason},
            "unsupported_claims": {"passed": c3_pass, "claims_found": c3_claims, "reason": c3_reason},
            "format_compliance": {"passed": c4_pass, "reason": c4_reason},
            "answer_sufficiency": {"passed": c5_pass, "reason": c5_reason},
            "appropriate_refusal": {"passed": c6_pass, "reason": c6_reason},
        }
    }
