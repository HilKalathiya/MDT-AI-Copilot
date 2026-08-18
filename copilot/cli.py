"""
copilot/cli.py — Plain CLI chat loop for the MDT AI Copilot.

Usage:
    python -m copilot.cli
    python -m copilot.cli --verbose     # shows tool calls
    python -m copilot.cli --no-color    # plain text output

Phase 3 demo questions (from Section 7.6):
  1. "How many MDT samples do we have for cell 1?"           (structured data)
  2. "Which UEs show anomalous RSRP patterns?"               (structured + ML)
  3. "Why does the gNB use a ring buffer instead of a plain array?"  (conceptual)
  4. "What is the RSRP threshold used for the rsrp_drop trigger?"    (conceptual)
  5. "Cell 1 has anomalies — is that expected given MDT trigger logic?" (hybrid)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ANSI colors (disabled with --no-color)
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         MDT AI Copilot  —  5G Coverage Analyst          ║
║         Powered by LangGraph + Cohere Command-R        ║
╚══════════════════════════════════════════════════════════╝
  Type your question and press Enter.
  Commands: 'exit' | 'quit' | 'help' | 'clear'
"""

HELP_TEXT = """
Available commands:
  exit / quit     — Exit the copilot
  help            — Show this help message
  clear           — Clear the screen
  verbose on/off  — Toggle tool-call trace output

Example questions:
  • How many MDT samples do we have for cell 1?
  • Which UEs show anomalous RSRP patterns?
  • Why does the gNB use a ring buffer instead of a plain array?
  • What RSRP threshold triggers the rsrp_drop reason?
  • Cell 1 has anomalies — is that expected given MDT trigger logic?
  • What does the suggest_threshold tool recommend for cell 1?
"""


def run_cli(verbose: bool = False, no_color: bool = False) -> None:
    c = lambda code, text: (text if no_color else f"{code}{text}{RESET}")

    print(c(CYAN, BANNER))
    print(c(YELLOW, "  Loading agent (first query may take a moment)...\n"))

    from copilot.agent import ask

    print(c(GREEN, "  ✅  Agent ready. Ask away!\n"))

    while True:
        try:
            raw = input(c(BOLD, "You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(c(YELLOW, "\n\nGoodbye!"))
            break

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("exit", "quit"):
            print(c(YELLOW, "Goodbye!"))
            break
        elif cmd == "help":
            print(c(CYAN, HELP_TEXT))
            continue
        elif cmd == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(c(CYAN, BANNER))
            continue
        elif cmd == "verbose on":
            verbose = True
            print(c(YELLOW, "  Verbose mode ON — tool calls will be shown.\n"))
            continue
        elif cmd == "verbose off":
            verbose = False
            print(c(YELLOW, "  Verbose mode OFF.\n"))
            continue

        print(c(YELLOW, "\nCopilot: ") + c(CYAN, "Thinking...\n"))
        try:
            answer = ask(raw, verbose=verbose)
            print(c(GREEN, "Copilot: ") + answer + "\n")
        except Exception as e:
            print(c(YELLOW, f"  ⚠️  Error: {e}\n"))
            if verbose:
                import traceback
                traceback.print_exc()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="MDT AI Copilot — CLI chat")
    parser.add_argument("--verbose", action="store_true",
                        help="Show tool call trace for each question")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color output")
    args = parser.parse_args()
    run_cli(verbose=args.verbose, no_color=args.no_color)


if __name__ == "__main__":
    _cli()
