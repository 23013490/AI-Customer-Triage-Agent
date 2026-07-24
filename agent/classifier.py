"""
classifier.py — Domain, product-area, and request-type classification.

Uses keyword heuristics + retrieval scores to determine:
- Which company/domain this ticket belongs to
- The product area within that domain
- The request type (bug, feature_request, product_issue, invalid)
"""

from __future__ import annotations

import re
from typing import Optional, List, Tuple
from .corpus import RetrievalResult


# ---------------------------------------------------------------------------
# Domain / product-area taxonomy
# ---------------------------------------------------------------------------

# Maps (company, keyword patterns) → product area label
PRODUCT_AREA_RULES: List[Tuple[str, str, str]] = [
    # ── HackerRank ──────────────────────────────────────────────────────────
    ("HackerRank", r"\b(assessment|test|coding test|screening|invite|candidate)\b",
     "Assessments"),
    ("HackerRank", r"\b(interview|live interview|video interview|pairing)\b",
     "Interviews"),
    ("HackerRank", r"\b(hackathon|contest|competition|leaderboard|rank)\b",
     "Contests & Hackathons"),
    ("HackerRank", r"\b(question|challenge|problem|submission|solution|code editor)\b",
     "Challenges & Submissions"),
    ("HackerRank", r"\b(report|analytics|score|result|report card)\b",
     "Reports & Analytics"),
    ("HackerRank", r"\b(billing|invoice|subscription|plan|payment|pricing)\b",
     "Billing & Subscriptions"),
    ("HackerRank", r"\b(account|login|password|2fa|sso|saml|api key)\b",
     "Account & Authentication"),
    ("HackerRank", r"\b(integration|ats|greenhouse|lever|workday|webhook|api)\b",
     "Integrations & API"),
    ("HackerRank", r"\b(plagiarism|cheating|integrity|proctoring|webcam)\b",
     "Assessment Integrity"),
    ("HackerRank", r"\b(candidate experience|feedback|email|notification)\b",
     "Candidate Experience"),

    # ── Claude ──────────────────────────────────────────────────────────────
    ("Claude", r"\b(api|rate limit|token|context window|model|claude.?\d)\b",
     "API & Developer"),
    ("Claude", r"\b(billing|invoice|subscription|plan|payment|credit|pro plan)\b",
     "Billing & Subscriptions"),
    ("Claude", r"\b(account|login|password|email|2fa|authentication)\b",
     "Account & Authentication"),
    ("Claude", r"\b(conversation|chat|memory|history|artifact|project)\b",
     "Conversations & Projects"),
    ("Claude", r"\b(content (policy|moderation|filter|block|refus))\b",
     "Content Policy"),
    ("Claude", r"\b(privacy|data|gdpr|training data|opt.?out)\b",
     "Privacy & Data"),
    ("Claude", r"\b(mobile|app|ios|android|desktop)\b",
     "Apps & Clients"),
    ("Claude", r"\b(upload|file|attachment|document|pdf|image)\b",
     "File Handling"),
    ("Claude", r"\b(slow|latency|down|outage|error|bug|crash|not working)\b",
     "Performance & Reliability"),
    ("Claude", r"\b(team|enterprise|workspace|admin|seats|organization)\b",
     "Teams & Enterprise"),

    # ── Visa ────────────────────────────────────────────────────────────────
    ("Visa", r"\b(fraud|unauthorized|dispute|chargeback|stolen)\b",
     "Fraud & Disputes"),
    ("Visa", r"\b(card (not working|declined|blocked|deactivated|replacement))\b",
     "Card Issues"),
    ("Visa", r"\b(contactless|nfc|tap to pay|apple pay|google pay|digital wallet)\b",
     "Digital Payments"),
    ("Visa", r"\b(international|foreign (transaction|currency)|travel|abroad)\b",
     "International Transactions"),
    ("Visa", r"\b(atm|cash (advance|withdrawal)|pin)\b",
     "ATM & Cash"),
    ("Visa", r"\b(reward|point|benefit|cashback|offer)\b",
     "Rewards & Benefits"),
    ("Visa", r"\b(bank|issuer|statement|account number)\b",
     "Issuing Bank Support"),
    ("Visa", r"\b(merchant|payment gateway|pos|terminal|acquiring)\b",
     "Merchant Services"),
    ("Visa", r"\b(virtual card|prepaid|gift card)\b",
     "Prepaid & Virtual Cards"),
    ("Visa", r"\b(3d secure|visa secure|otp|authentication)\b",
     "Authentication & Security"),
]

# ── Request-type rules ───────────────────────────────────────────────────────

REQUEST_TYPE_RULES: List[Tuple[str, str]] = [
    # bug
    (r"\b(bug|glitch|error|broken|crash|not working|fails|freezes|incorrect(ly)?|wrong (result|output|behavior))\b",
     "bug"),
    # feature_request
    (r"\b(feature (request|suggestion)|can you (add|support|implement)|would (be|love to) (have|see)|wish (there was|you had)|please (add|implement|support))\b",
     "feature_request"),
    # invalid — nonsense, OOS, injection
    (r"\b(ignore (previous|all)|jailbreak|DAN mode|pretend you are|act as)\b",
     "invalid"),
]

# Fallback product areas per company
DEFAULT_PRODUCT_AREA = {
    "HackerRank": "General Support",
    "Claude":     "General Support",
    "Visa":       "General Support",
    "None":       "General Support",
    "unknown":    "General Support",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_domain(
    issue: str,
    subject: str,
    declared_company: str,
    retrieval_results: List[RetrievalResult],
) -> str:
    """
    Determine the most likely company domain.
    Prefers the declared company; falls back to retrieval signal.
    """
    if declared_company and declared_company.strip().lower() not in ("none", "", "unknown"):
        return declared_company.strip()

    # Fall back to best retrieval hit
    if retrieval_results:
        return retrieval_results[0].chunk.domain
    return "unknown"


def classify_product_area(
    issue: str,
    subject: str,
    company: str,
    retrieval_results: Optional[List[RetrievalResult]] = None,
) -> str:
    """
    Return the product area for a ticket given its company.
    Tries keyword rules first; falls back to retrieval domain label.
    """
    combined = f"{subject} {issue}".lower()

    # Try company-specific rules first
    for rule_company, pattern, area in PRODUCT_AREA_RULES:
        if rule_company == company and re.search(pattern, combined, re.IGNORECASE):
            return area

    # Try any-company rules (useful when company = "None")
    if company in ("None", "unknown", ""):
        for _, pattern, area in PRODUCT_AREA_RULES:
            if re.search(pattern, combined, re.IGNORECASE):
                return area

    # Fall back to retrieval-based area
    if retrieval_results:
        top = retrieval_results[0].chunk
        return f"{top.domain} — {top.title[:50]}"

    return DEFAULT_PRODUCT_AREA.get(company, "General Support")


def classify_request_type(issue: str, subject: str) -> str:
    """
    Classify the ticket as: bug | feature_request | product_issue | invalid
    """
    combined = f"{subject} {issue}".lower()

    # Check explicit overrides first
    for pattern, rtype in REQUEST_TYPE_RULES:
        if re.search(pattern, combined, re.IGNORECASE):
            return rtype

    # Default: it's a product_issue (question / complaint / how-to)
    return "product_issue"
