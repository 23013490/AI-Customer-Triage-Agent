"""
pipeline.py — Orchestration layer.

Reads the input CSV, runs each ticket through the full agent pipeline,
and writes the output CSV.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, List, Optional

from .corpus      import CorpusIndex
from .classifier  import classify_domain, classify_product_area, classify_request_type
from .safety      import assess_safety, is_out_of_scope, is_injection
from .llm         import (
    generate_response,
    generate_escalation_message,
    generate_out_of_scope_message,
)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = ["status", "product_area", "response", "justification", "request_type"]
INPUT_FIELDS  = ["issue", "subject", "company"]


# ---------------------------------------------------------------------------
# Single-ticket processing
# ---------------------------------------------------------------------------

def process_ticket(
    issue: str,
    subject: str,
    company: str,
    index: CorpusIndex,
    verbose: bool = False,
) -> Dict[str, str]:
    """Full agent pipeline for one ticket."""

    t0 = time.time()

    # ── 0. Sanitise inputs ──────────────────────────────────────────────────
    issue   = (issue   or "").strip()
    subject = (subject or "").strip()
    company = (company or "None").strip()

    if not issue:
        return {
            "status":       "replied",
            "product_area": "General Support",
            "response":     "It looks like your message was empty. Please describe your issue and we will be happy to help.",
            "justification":"Empty ticket — no actionable content.",
            "request_type": "invalid",
        }

    # ── 1. Injection / manipulation check ──────────────────────────────────
    if is_injection(issue, subject):
        return {
            "status":       "escalated",
            "product_area": "Security",
            "response":     "We're unable to process this request. It has been flagged for security review.",
            "justification":"Detected possible prompt injection or policy manipulation attempt.",
            "request_type": "invalid",
        }

    # ── 2. Out-of-scope quick check ─────────────────────────────────────────
    oos = is_out_of_scope(issue, subject)

    # ── 3. Retrieval ────────────────────────────────────────────────────────
    # Build a rich query: subject + first 300 chars of issue
    query = f"{subject} {issue[:300]}"

    # Try domain-scoped retrieval first; if weak, try global
    domain_filter = company if company not in ("None", "unknown", "") else None
    results = index.retrieve(query, top_k=6, domain_filter=domain_filter)
    top_score = results[0].score if results else 0.0

    if top_score < 0.05 and domain_filter:
        # Widen to full corpus
        results = index.retrieve(query, top_k=6, domain_filter=None)
        top_score = results[0].score if results else 0.0

    # ── 4. Classify ─────────────────────────────────────────────────────────
    resolved_company = classify_domain(issue, subject, company, results)
    product_area     = classify_product_area(issue, subject, resolved_company, results)
    request_type     = classify_request_type(issue, subject)

    if oos:
        request_type = "invalid"

    # ── 5. Safety assessment ─────────────────────────────────────────────────
    safety = assess_safety(issue, subject, resolved_company, top_score)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  Subject  : {subject[:80]}")
        print(f"  Company  : {company} → {resolved_company}")
        print(f"  Area     : {product_area}")
        print(f"  ReqType  : {request_type}")
        print(f"  Risk     : {safety.risk_level}  escalate={safety.should_escalate}")
        print(f"  TopScore : {top_score:.4f}")
        print(f"  OOS      : {oos}")
        if safety.risk_reasons:
            print(f"  Reasons  : {'; '.join(safety.risk_reasons)}")

    # ── 6. Route: escalate or reply ──────────────────────────────────────────
    if oos and not safety.should_escalate:
        # Out-of-scope but not dangerous → reply politely
        status   = "replied"
        response = generate_out_of_scope_message(issue, resolved_company)
        justification = (
            "Issue is outside the scope of HackerRank, Claude, or Visa support. "
            "Replied with an out-of-scope notice."
        )

    elif safety.should_escalate:
        status   = "escalated"
        response = generate_escalation_message(
            issue, resolved_company, product_area, safety.risk_reasons
        )
        justification = (
            f"Escalated due to: {'; '.join(safety.risk_reasons)}. "
            f"Risk level: {safety.risk_level}. "
            "Requires human agent review."
        )

    else:
        # Safe to attempt a grounded reply
        status   = "replied"
        response = generate_response(
            issue, subject, resolved_company, product_area, results
        )
        justification = _build_justification(
            resolved_company, product_area, results, top_score, safety
        )

    elapsed = time.time() - t0
    if verbose:
        print(f"  Status   : {status}  ({elapsed:.1f}s)")

    return {
        "status":       status,
        "product_area": product_area,
        "response":     response,
        "justification":justification,
        "request_type": request_type,
    }


def _build_justification(
    company: str,
    product_area: str,
    results,
    top_score: float,
    safety,
) -> str:
    source_note = ""
    if results:
        top = results[0]
        source_note = f"Top corpus match: '{top.chunk.title[:60]}' (score={top_score:.3f})."
    coverage = (
        "Good corpus coverage." if top_score >= 0.10
        else "Limited corpus coverage — response may be partial."
    )
    return (
        f"Classified as {company} / {product_area}. "
        f"{source_note} {coverage} "
        f"Risk level: {safety.risk_level}."
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    output_path: str,
    corpus_dir: str,
    verbose: bool = False,
):
    # ── Load corpus ──────────────────────────────────────────────────────────
    print(f"[1/3] Loading corpus from: {corpus_dir}")
    index = CorpusIndex()
    try:
        n = index.load_directory(corpus_dir)
        stats = index.stats
        print(f"      Loaded {n} files → {stats['total_chunks']} chunks")
        for domain, count in stats["by_domain"].items():
            print(f"        {domain}: {count} chunks")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Continuing without corpus — responses will be limited.")

    # ── Read input ───────────────────────────────────────────────────────────
    print(f"\n[2/3] Reading tickets from: {input_path}")
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    rows = []
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"      Found {len(rows)} tickets")

    # ── Process each ticket ──────────────────────────────────────────────────
    print(f"\n[3/3] Processing tickets → {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    out_rows = []
    for i, row in enumerate(rows, 1):
        issue   = row.get("issue",   "") or ""
        subject = row.get("subject", "") or ""
        company = row.get("company", "") or "None"

        if verbose:
            print(f"\n[{i}/{len(rows)}]", end="")
        else:
            print(f"\r  Processing {i}/{len(rows)}...", end="", flush=True)

        result = process_ticket(issue, subject, company, index, verbose=verbose)

        # Merge input + output into one row for full output
        out_row = {k: row.get(k, "") for k in row}
        out_row.update(result)
        out_rows.append(out_row)

    if not verbose:
        print()

    # ── Write output ──────────────────────────────────────────────────────────
    if not out_rows:
        print("No rows to write.")
        return

    # Determine output field ordering: input fields first, then output fields
    all_keys = list(rows[0].keys()) + [f for f in OUTPUT_FIELDS if f not in rows[0]]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n✓ Done. Output written to: {output_path}")
    _print_summary(out_rows)


def _print_summary(rows: List[Dict]):
    total      = len(rows)
    escalated  = sum(1 for r in rows if r.get("status") == "escalated")
    replied    = sum(1 for r in rows if r.get("status") == "replied")
    by_type: Dict[str, int] = {}
    for r in rows:
        t = r.get("request_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\n{'─'*50}")
    print(f"  Summary")
    print(f"{'─'*50}")
    print(f"  Total tickets : {total}")
    print(f"  Replied       : {replied} ({replied/total*100:.0f}%)")
    print(f"  Escalated     : {escalated} ({escalated/total*100:.0f}%)")
    print(f"  Request types : {by_type}")
    print(f"{'─'*50}")
