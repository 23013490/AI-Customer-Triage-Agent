#!/usr/bin/env python3
"""
Multi-Domain Support Triage Agent
Entry point: reads support_tickets.csv / support_issues.csv, runs agent, writes output.csv
"""

import sys
import os
import argparse
from agent.pipeline import run_pipeline


def _find_file(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            return path
    print(f"WARNING: Could not auto-detect {label}. Tried: {candidates}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent")
    parser.add_argument("--input",   default=None, help="Path to input CSV")
    parser.add_argument("--output",  default=None, help="Path to output CSV")
    parser.add_argument("--sample",  action="store_true", help="Run on sample tickets")
    parser.add_argument("--corpus",  default="../data", help="Corpus directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.sample:
        args.input = _find_file([
            "../support_tickets/sample_support_tickets.csv",
            "../support_issues/sample_support_tickets.csv",
            "../support_issues/sample_support_issues.csv",
        ], "sample CSV")
        args.output = os.path.join(os.path.dirname(args.input), "sample_output.csv")
    else:
        if args.input is None:
            args.input = _find_file([
                "../support_tickets/support_tickets.csv",
                "../support_issues/support_issues.csv",
            ], "support CSV")
        if args.output is None:
            args.output = os.path.join(os.path.dirname(args.input), "output.csv")

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        corpus_dir=args.corpus,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
