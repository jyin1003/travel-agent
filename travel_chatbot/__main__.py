"""
travel_chatbot/__main__.py

Entry point for the Travel Chatbot CLI.
"""

from __future__ import annotations

import argparse
import json
import logging, warnings
import sys, os


# ---------------------------------------------------------------------------
# Parse args BEFORE importing the agent so we can configure logging first
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="travel_chatbot",
        description="Personal travel knowledge-base chatbot.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable full logging and print the complete agent state after each turn.",
    )
    return parser.parse_args()


args = _parse_args()

# ---------------------------------------------------------------------------
# Logging configuration - happens before any agent import
# ---------------------------------------------------------------------------

if args.profile:
    # Re-use the same format as nodes.py so log lines look consistent
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [NODES] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,          # keep logs on stderr so stdout stays clean
    )
    # In profile mode: do NOT install any print/stream filters — logging
    # internals must have direct, unmodified access to sys.stderr.
else:
    # Silence everything – set root logger to WARNING and kill any existing handlers
    logging.basicConfig(level=logging.WARNING)
    for name in list(logging.Logger.manager.loggerDict):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Also suppress LangChain / LangGraph internal loggers that configure themselves
    for noisy in ("langchain", "langchain_core", "langgraph", "httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Suppress model warnings from HF (only needed in silent mode)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    logging.getLogger("mlx_lm").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    os.environ["TQDM_DISABLE"] = "1"

    import builtins

    _real_print = builtins.print

    class _FilteredStream:
        _SUPPRESS = ("LOAD REPORT", "UNEXPECTED", "Loading weights", "Notes:", "---", "Key ")

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def write(self, msg):
            if any(kw in msg for kw in self._SUPPRESS):
                return len(msg)
            return self._wrapped.write(msg)

        def flush(self, *args, **kwargs):
            return self._wrapped.flush(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def _filtered_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if any(kw in msg for kw in _FilteredStream._SUPPRESS):
            return
        _real_print(*args, **kwargs)

    builtins.print = _filtered_print
    sys.stderr = _FilteredStream(sys.stderr)

# Now safe to import the agent
from agent.graph import run_query  # noqa: E402  (import after logging setup)


# ---------------------------------------------------------------------------
# ANSI helpers (degrade gracefully on terminals that don't support colour)
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

BOLD   = lambda t: _c("1",     t)
CYAN   = lambda t: _c("1;36",  t)
GREEN  = lambda t: _c("1;32",  t)
YELLOW = lambda t: _c("1;33",  t)
DIM    = lambda t: _c("2",     t)


# ---------------------------------------------------------------------------
# Intro banner
# ---------------------------------------------------------------------------

BANNER = f"""
{CYAN('╔══════════════════════════════════════════════════════╗')}
{CYAN('║')}  {BOLD('✈  Personal Travel Chatbot')}                          {CYAN('║')}
{CYAN('║')}  {DIM('Powered by your personal travel knowledge base')}      {CYAN('║')}
{CYAN('╚══════════════════════════════════════════════════════╝')}

Ask me anything about your travels — past trips, spending,
photos, places you've saved, or where to go next.

{DIM("Type 'exit' or 'quit' to leave.")}
{DIM("Run with --profile for full debug output.")}
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print(BANNER)

    if args.profile:
        print(YELLOW("[ profile mode: full logging + state dump enabled ]\n"))

    session: dict = {}

    while True:
        try:
            user_input = input(BOLD("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Safe travels. ✈")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! Safe travels. ✈")
            break

        result = run_query(user_input, session_memory=session)
        session.update(result.get("memory", {}))

        answer = result.get("answer", "").strip()
        print(f"\n{GREEN('Travel Chatbot:')} {answer}\n")

        # --profile: dump full state to stderr so it doesn't mix with the answer
        if args.profile:
            state_dump = {
                k: v for k, v in result.items()
                if k != "messages"          # BaseMessage objects aren't JSON-serialisable by default
            }
            print(
                YELLOW("\n── full agent state ──────────────────────────────────\n")
                + json.dumps(state_dump, indent=2, default=str)
                + YELLOW("\n──────────────────────────────────────────────────────\n"),
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()