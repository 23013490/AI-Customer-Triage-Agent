# Multi-Domain Support Triage Agent

A terminal-based agent that triages support tickets across **HackerRank**, **Claude (Anthropic)**, and **Visa** using only the provided local corpus — no external knowledge.

## Architecture

```
main.py                      ← CLI entry point
agent/
  corpus.py                  ← Corpus loader + TF-IDF retrieval index
  classifier.py              ← Domain, product-area & request-type classification
  safety.py                  ← Risk assessment & escalation logic
  llm.py                     ← Claude API call for response generation
  pipeline.py                ← Orchestration: reads CSV → processes → writes CSV
```

### Pipeline (per ticket)

```
Input ticket
    │
    ▼
[Injection / manipulation check]  ──→ invalid + escalate
    │
    ▼
[Out-of-scope detection]           ──→ replied (out-of-scope message)
    │
    ▼
[TF-IDF Retrieval] from local corpus (domain-scoped first, then global)
    │
    ▼
[Classification]
  · Domain (HackerRank / Claude / Visa)
  · Product Area (Assessments, Billing, Fraud & Disputes, …)
  · Request Type (bug / feature_request / product_issue / invalid)
    │
    ▼
[Safety Assessment]
  · Keyword signals: fraud, legal threats, account compromise, …
  · Corpus coverage score
  · Urgency: routine / urgent / critical
    │
    ├── High/Critical risk ──→ escalated (template message)
    │
    └── Low/Medium risk   ──→ Claude API generates grounded response
                                (falls back to corpus excerpt if API unavailable)
    │
    ▼
Output row: status, product_area, response, justification, request_type
```

## Usage

```bash
cd code

# Run on the main ticket file
python main.py

# Run on sample tickets (for validation)
python main.py --sample

# Custom paths
python main.py --input ../support_tickets/support_tickets.csv \
               --output ../support_tickets/output.csv \
               --corpus ../data

# Verbose mode (shows per-ticket reasoning)
python main.py --verbose
```

## Output fields

| Field          | Values                                          |
|----------------|-------------------------------------------------|
| `status`       | `replied` \| `escalated`                        |
| `product_area` | Domain-specific area (e.g. "Fraud & Disputes")  |
| `response`     | User-facing reply, grounded in corpus           |
| `justification`| Internal reasoning for the decision             |
| `request_type` | `product_issue` \| `bug` \| `feature_request` \| `invalid` |

## Escalation logic

The agent escalates when it detects:
- **Fraud or unauthorized transactions** (critical)
- **Account compromise / identity theft** (critical)
- **Legal threats or law enforcement** (critical)
- **Data privacy / GDPR violations** (high)
- **Physical safety concerns** (critical)
- **Billing disputes** with low corpus coverage (medium+)
- **Prompt injection attempts** (always escalated as invalid)

## Retrieval

Uses TF-IDF cosine similarity over chunked corpus text. No external libraries needed — pure Python stdlib math. Corpus is loaded once at startup and reused for all tickets.

## Requirements

- Python 3.8+
- No external packages required
- Anthropic API key must be set (handled by the Claude.ai environment)
