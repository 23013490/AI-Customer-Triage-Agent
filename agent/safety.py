"""
safety.py — Risk and safety classification for support tickets.

Determines whether a ticket should be escalated based on:
- Topic sensitivity (fraud, legal, billing disputes, account security, etc.)
- Toxicity / manipulation / prompt injection attempts
- Insufficient corpus coverage to answer safely
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SafetyAssessment:
    should_escalate: bool
    risk_level: str          # low | medium | high | critical
    risk_reasons: List[str]
    urgency: str             # routine | urgent | critical


# ---------------------------------------------------------------------------
# Keyword signal tables
# ---------------------------------------------------------------------------

# These patterns in the ticket text strongly suggest escalation is needed.
ESCALATION_SIGNALS: List[Tuple[str, str, str]] = [
    # (regex pattern, risk_level, reason label)

    # Financial fraud / account takeover
    (r"\bfraud\b|\bfraudulent\b|\bunauthorized (charge|transaction|access)\b",
     "critical", "Potential fraud or unauthorized financial activity"),
    (r"\bstolen\b.*\bcard\b|\bcard\b.*\bstolen\b", "critical", "Stolen card report"),
    (r"\bidentity theft\b|\baccount (hacked|compromised|taken over)\b",
     "critical", "Identity theft or account compromise"),
    (r"\bdispute\b.*\btransaction\b|\bchargeback\b",
     "high", "Transaction dispute requiring manual review"),

    # Legal / compliance
    (r"\blegal action\b|\blawsuit\b|\blitigate\b|\battorney\b|\blawyer\b",
     "high", "Legal threat or legal inquiry"),
    (r"\bGDPR\b|\bdata (breach|leak)\b|\bprivacy (violation|complaint)\b",
     "high", "Data privacy or regulatory concern"),
    (r"\blaw enforcement\b|\bsubpoena\b|\bcourt order\b",
     "critical", "Law enforcement or legal order"),

    # Physical safety
    (r"\bsuicid(e|al)\b|\bself.harm\b|\bhurt (myself|yourself)\b",
     "critical", "User safety concern"),
    (r"\bthreat(en)?\b.*\b(violence|harm|kill)\b",
     "critical", "Threat of violence"),

    # Billing / payment sensitive
    (r"\brefund\b.*\b(denied|reject|not received)\b|\bovercharg(ed|ing)\b",
     "medium", "Billing dispute or refund issue"),
    (r"\bdouble.?charg(ed|ing)\b|\bincorrect (amount|charge|billing)\b",
     "medium", "Incorrect billing"),

    # Account access / locked out
    (r"\b(locked out|cannot (login|log in|access)|account (locked|suspended|banned|disabled))\b",
     "medium", "Account access issue requiring verification"),
    (r"\bpassword (reset|recovery)\b|\b2fa\b|\btwo.factor\b",
     "medium", "Account security / authentication issue"),

    # Assessment / exam integrity (HackerRank specific)
    (r"\b(cheating|plagiarism|academic (dishonesty|integrity)|impersonation)\b",
     "high", "Academic integrity concern"),
    (r"\btest (result|score) (wrong|incorrect|unfair|disputed)\b",
     "medium", "Disputed assessment result"),
]

# Prompt injection / manipulation attempts
INJECTION_PATTERNS = [
    r"ignore (previous|all|your) instructions",
    r"you are now",
    r"disregard (your|the|all) (rules|instructions|guidelines|training)",
    r"pretend (you are|to be)",
    r"act as (if you are|a|an)",
    r"jailbreak",
    r"DAN mode",
    r"override (your|all) (safety|guidelines|instructions)",
    r"reveal (your|the) (system|hidden) (prompt|instructions)",
    r"forget (everything|all) (you know|you were told)",
]

# Topics completely outside all three domains
OUT_OF_SCOPE_PATTERNS = [
    r"\b(recipe|food|cook|restaurant)\b",
    r"\b(weather|forecast|temperature)\b",
    r"\b(sports|football|cricket|basketball)\b",
    r"\b(movie|film|tv show|series|netflix|youtube)\b",
    r"\b(relationship advice|dating|marriage)\b",
    r"\b(medical advice|diagnos(e|is)|symptoms|medication|doctor)\b",
    r"\b(astrology|horoscope|zodiac)\b",
    r"\b(homework|essay|write (my|an) essay)\b",
    r"\b(stock (market|price|tip)|cryptocurrency|bitcoin|invest(ing|ment) advice)\b",
]


def assess_safety(
    issue: str,
    subject: str,
    company: str,
    retrieval_score: float = 0.0,
) -> SafetyAssessment:
    """
    Evaluate whether a ticket should be escalated.

    Parameters
    ----------
    issue           : ticket body text
    subject         : ticket subject line
    company         : declared company domain
    retrieval_score : best TF-IDF cosine score from corpus retrieval
    """
    combined = f"{subject} {issue}".lower()
    risk_reasons: List[str] = []
    max_risk = "low"
    urgency  = "routine"

    # 1. Check for prompt injection / manipulation
    for pat in INJECTION_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            risk_reasons.append("Possible prompt injection or manipulation attempt")
            max_risk = _escalate_risk(max_risk, "high")
            break

    # 2. Check escalation signals
    for pat, risk, reason in ESCALATION_SIGNALS:
        if re.search(pat, combined, re.IGNORECASE):
            risk_reasons.append(reason)
            max_risk = _escalate_risk(max_risk, risk)

    # 3. Check out-of-scope
    oos_hits = []
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            oos_hits.append(pat)
    if oos_hits:
        risk_reasons.append("Issue appears out of scope for all three support domains")
        # Out-of-scope doesn't necessarily escalate — agent will reply "out of scope"

    # 4. Low corpus coverage → escalate if medium+ risk
    if retrieval_score < 0.05 and not oos_hits:
        risk_reasons.append(
            "Insufficient corpus coverage to answer this query confidently"
        )
        max_risk = _escalate_risk(max_risk, "medium")

    # 5. Determine urgency
    if max_risk == "critical":
        urgency = "critical"
    elif max_risk == "high":
        urgency = "urgent"
    elif max_risk == "medium":
        urgency = "urgent"

    # 6. Decide escalation
    should_escalate = max_risk in ("high", "critical") or (
        max_risk == "medium" and retrieval_score < 0.08
    )

    return SafetyAssessment(
        should_escalate=should_escalate,
        risk_level=max_risk,
        risk_reasons=risk_reasons,
        urgency=urgency,
    )


def is_out_of_scope(issue: str, subject: str) -> bool:
    combined = f"{subject} {issue}".lower()
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            return True
    return False


def is_injection(issue: str, subject: str) -> bool:
    combined = f"{subject} {issue}".lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _escalate_risk(current: str, new: str) -> str:
    if _RISK_ORDER.get(new, 0) > _RISK_ORDER.get(current, 0):
        return new
    return current
