"""
travel_chatbot/__main__.py

Entry point for the Travel Chatbot CLI.

Image input: append image file paths anywhere in your message and they will
be automatically detected and passed to the agent as visual context.

  Example:
    You: what's in this photo? /photos/tokyo_dinner.jpg
    You: describe these two shots /a.jpg /b.png and tell me where I was
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import warnings
import sys
import os


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
# Logging configuration — happens before any agent import
# ---------------------------------------------------------------------------

if args.profile:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [NODES] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
else:
    logging.basicConfig(level=logging.WARNING)
    for name in list(logging.Logger.manager.loggerDict):
        logging.getLogger(name).setLevel(logging.WARNING)
    for noisy in ("langchain", "langchain_core", "langgraph", "httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

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
from agent.graph import run_query  # noqa: E402


# ---------------------------------------------------------------------------
# ANSI helpers
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
# Image path parsing
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _parse_input(raw: str) -> tuple[str, list[str]]:
    """
    Split a user input line into (query_text, image_paths).

    Tokens ending in a known image extension are extracted as image paths;
    the remainder is returned as the query text.

    Uses shlex.split so quoted paths with spaces work:
        'what is this? "/my photos/tokyo dinner.jpg"'

    Falls back to whitespace split if shlex fails (e.g. unmatched quotes).

    Examples:
        "what's in this? /photos/abc.jpg"
            → ("what's in this?", ["/photos/abc.jpg"])
        "compare these /a.jpg /b.png and tell me where I was"
            → ("compare these  and tell me where I was", ["/a.jpg", "/b.png"])
        "just a normal question"
            → ("just a normal question", [])
    """
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()

    image_paths: list[str] = []
    text_tokens: list[str] = []

    for token in tokens:
        if any(token.lower().endswith(ext) for ext in _IMAGE_EXTS):
            image_paths.append(token)
        else:
            text_tokens.append(token)

    query_text = " ".join(text_tokens).strip()
    # If everything was an image path and no query text remains, use the
    # original raw input as the query so the agent still has something to work with
    if not query_text:
        query_text = raw.strip()

    return query_text, image_paths


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

You can include image file paths in your message:
  {DIM('e.g.  what restaurant is this? /photos/dinner.jpg')}

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

        query_text, image_paths = _parse_input(user_input)

        # Warn the user about any paths that don't exist on disk
        missing = [p for p in image_paths if not os.path.exists(p)]
        for p in missing:
            print(DIM(f"  ⚠  Image not found: {p}"))
        image_paths = [p for p in image_paths if os.path.exists(p)]

        if image_paths:
            print(DIM(f"  [attaching {len(image_paths)} image(s): {', '.join(os.path.basename(p) for p in image_paths)}]"))

        result = run_query(query_text, session_memory=session, image_paths=image_paths)
        new_memory = result.get("memory", {})
        if new_memory:
            session.update(new_memory)

        answer = result.get("answer", "").strip()
        print(f"\n{GREEN('Travel Chatbot:')} {answer}\n")

        if args.profile:
            state_dump = {
                k: v for k, v in result.items()
                if k != "messages"
            }
            print(
                YELLOW("\n── full agent state ──────────────────────────────────\n")
                + json.dumps(state_dump, indent=2, default=str)
                + YELLOW("\n──────────────────────────────────────────────────────\n"),
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()