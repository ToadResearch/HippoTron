# https://github.com/ZHZisZZ/dllm/blob/main/dllm/utils/visualizers.py
# https://github.com/ZHZisZZ/dllm/blob/main/dllm/utils/chat.py
import argparse
import logging
import os
import shutil
import sys
import time
import warnings
import textwrap
from typing import Tuple


def colored(st, color: str | None, background: bool = False) -> str:
    """Lightweight color helper (avoids termcolor dependency)."""
    colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    return (
        f"\u001b[{10 * background + 60 * (color.upper() == color) + 30 + colors.index(color.lower())}m{st}\u001b[0m"
        if color is not None
        else st
    )


# Silence noisy warnings/logs
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure local modules are importable
ROOT = os.path.dirname(os.path.abspath(__file__))
HIPPO_PATH = os.path.join(ROOT, "hippotron")
if HIPPO_PATH not in sys.path:
    sys.path.append(HIPPO_PATH)

from config import Config  # type: ignore  # noqa: E402
from model import BERTdLLM  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HippoTron masked LLM CLI")
    parser.add_argument(
        "-m",
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate (overrides config for this session)",
    )
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggle animated step-by-step display of generation",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output",
    )
    return parser.parse_args()


def count_prompt_tokens(model: BERTdLLM, text: str, config: Config) -> Tuple[int, int]:
    """Return (used_prompt_len, raw_prompt_len)."""
    encoded = model.tokenizer(
        text,
        max_length=config.MAX_LEN,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )
    raw_len = encoded["input_ids"].shape[1]
    used_len = min(raw_len, config.MAX_PREFIX_LEN)
    return used_len, raw_len


def maybe_color(enabled: bool, text: str, color_name: str | None = None) -> str:
    return colored(text, color_name) if enabled and color_name else text


SPECIAL_COLOR_MAP = {
    "[MASK]": "yellow",
    "[SEP]": "cyan",
    "[CLS]": "magenta",
    "[PAD]": "blue",
    "[UNK]": "red",
}


def colorize_special_tokens(text: str, use_color: bool) -> str:
    if not use_color:
        return text
    out = text
    for token, col in SPECIAL_COLOR_MAP.items():
        out = out.replace(token, colored(token, col))
    return out


def strip_seed_prefix(text: str) -> str:
    return text[3:] if text.startswith("S] ") else text


def count_masks_in_text(text: str) -> int:
    return text.count("[MASK]")


def wrap_compact(text: str, width: int, max_lines: int = 8) -> list[str]:
    """Wrap text to width; if overflow, keep head+tail with ellipsis."""
    wrapped = textwrap.wrap(
        text,
        width=max(10, width),
        replace_whitespace=False,
        drop_whitespace=False,
    )
    if len(wrapped) <= max_lines:
        return wrapped if wrapped else [""]
    head = max_lines // 2
    tail = max_lines - head - 1
    return wrapped[:head] + ["…"] + wrapped[-tail:]


def print_help(use_color: bool) -> None:
    print(maybe_color(use_color, "\nCommands:", "cyan"))
    print("  :help              Show this help")
    print("  :quit / :exit      Quit the CLI")
    print("  :animate on|off    Toggle animated generation display")
    print("  :max <int>         Set max new tokens for subsequent generations\n")


def main() -> None:
    args = parse_args()
    use_color = not args.no_color

    config = Config()
    if args.max_new_tokens is not None:
        config.MAX_NEW_TOKENS = args.max_new_tokens

    model = BERTdLLM(config)
    animate = bool(args.animate)
    max_new_tokens = config.MAX_NEW_TOKENS

    print(maybe_color(use_color, "\nHippoTron Masked LLM", "magenta"))
    print(f"Model loaded from: {config.OUTPUT_DIR}")
    print("Type ':help' for commands, or enter text to generate.\n")

    while True:
        try:
            user_input = input(maybe_color(use_color, "prompt> ", "green"))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input.strip():
            continue

        lowered = user_input.strip().lower()
        if lowered in {"quit", "exit", ":quit", ":exit"}:
            print("Goodbye!")
            break
        if lowered == ":help":
            print_help(use_color)
            continue
        if lowered.startswith(":animate"):
            _, *rest = lowered.split()
            if rest and rest[0] in {"on", "off"}:
                animate = rest[0] == "on"
                state = "on" if animate else "off"
                print(f"Animation set to {state}.")
            else:
                print("Usage: :animate on|off")
            continue
        if lowered.startswith(":max"):
            _, *rest = lowered.split()
            if rest:
                try:
                    new_max = int(rest[0])
                    assert new_max > 0, "must be > 0"
                    assert new_max < config.MAX_LEN, f"must be < {config.MAX_LEN}"
                    max_new_tokens = new_max
                    print(f"Max new tokens set to {max_new_tokens}.")
                except (ValueError, AssertionError) as exc:
                    print(f"Invalid value for :max ({exc}).")
            else:
                print(f"Current max new tokens: {max_new_tokens}")
            continue

        used_len, raw_len = count_prompt_tokens(model, user_input, config)
        if raw_len > used_len:
            token_info = f"{used_len} tokens used (trimmed from {raw_len} to max prefix {config.MAX_PREFIX_LEN})"
        else:
            token_info = f"{used_len} tokens used"

        print(maybe_color(use_color, "\n[Prompt]", "cyan"))
        print(user_input)
        print(maybe_color(use_color, f"[Tokens] {token_info}", "cyan"))
        print()

        ids, history = model.generate(
            user_input,
            save_history=animate,
            max_new_tokens=max_new_tokens,
        )
        generated = strip_seed_prefix(model.postprocess(ids[0]))

        if animate and history:
            print(maybe_color(use_color, "[Animating generation]", "yellow"))
            term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
            total_steps = len(history)

            try:
                from rich.console import Console  # type: ignore
                from rich.live import Live  # type: ignore
                from rich.panel import Panel  # type: ignore
                from rich.progress import (  # type: ignore
                    Progress,
                    BarColumn,
                    TextColumn,
                    TimeRemainingColumn,
                    MofNCompleteColumn,
                    SpinnerColumn,
                )
                from rich.text import Text  # type: ignore
                from rich.layout import Layout  # type: ignore

                console = Console(force_terminal=True, color_system="truecolor", width=term_width)
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Diffusion"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TextColumn("[cyan]Masks: {task.fields[masks]}"),
                    TextColumn("•"),
                    TextColumn("[magenta]{task.fields[pct]:>4s}"),
                    TimeRemainingColumn(),
                    expand=True,
                )

                init_masks = count_masks_in_text(history[0]) if history else 0
                task_id = progress.add_task(
                    "Generating", total=total_steps, masks=init_masks, pct="0%"
                )

                layout = Layout()
                layout.split_column(
                    Layout(name="text", ratio=1),
                    Layout(name="progress", size=3),
                )

                refresh_hz = 24
                max_lines = 8

                with Live(layout, console=console, refresh_per_second=refresh_hz):
                    for step, partial in enumerate(history, start=1):
                        partial_clean = strip_seed_prefix(partial)
                        masks_remaining = count_masks_in_text(partial_clean)
                        pct = f"{int(100 * step / max(total_steps, 1))}%"
                        progress.update(task_id, advance=1, masks=masks_remaining, pct=pct)

                        colored_text = colorize_special_tokens(partial_clean, use_color)
                        wrapped_lines = wrap_compact(colored_text, width=term_width - 4, max_lines=max_lines)
                        text_block = "\n".join(wrapped_lines)
                        layout["text"].update(
                            Panel(
                                Text.from_ansi(text_block),
                                title=f"Step {step}/{total_steps}",
                                border_style="cyan",
                                padding=(1, 1),
                            )
                        )
                        layout["progress"].update(Panel(progress))
                        time.sleep(0.02)

                console.print()
            except Exception:
                # Fallback: overwrite a few wrapped lines in-place
                prev_lines = 0
                for step, partial in enumerate(history, start=1):
                    cleaned = strip_seed_prefix(partial.replace("\n", " ").strip())
                    text_only = colorize_special_tokens(cleaned, use_color)
                    wrapped = wrap_compact(text_only, width=max(20, term_width - 2), max_lines=6)
                    status = f"step {step}/{total_steps} • masks={count_masks_in_text(cleaned)} • {int(100*step/max(total_steps,1))}%"
                    status = maybe_color(use_color, status, "cyan")
                    frame_lines = wrapped + [status]

                    if prev_lines:
                        sys.stdout.write(f"\x1b[{prev_lines}A")

                    for _ in range(prev_lines):
                        sys.stdout.write("\r\033[K\n")
                    if prev_lines:
                        sys.stdout.write(f"\x1b[{prev_lines}A")

                    for line in frame_lines:
                        sys.stdout.write("\r\033[K" + line + "\n")
                    sys.stdout.flush()
                    prev_lines = len(frame_lines)
                    time.sleep(0.05)
                sys.stdout.write("\r\033[K")

        show_output = not (animate and history)
        if show_output:
            print(maybe_color(use_color, "[Output]", "cyan"))
            print(colorize_special_tokens(generated.strip(), use_color), end="\n\n")


if __name__ == "__main__":
    main()