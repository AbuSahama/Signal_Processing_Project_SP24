"""
utils.py

Small input-validation helpers for the command-line interface (main.py).

Note: this file wasn't provided in the original project, only referenced
by main.py's imports. This is an inferred implementation based on how
main.py calls each function (get_float_input(prompt, minimum=...),
get_int_input(prompt, minimum=...), get_choice(prompt, choices),
validate_filename(filename)) — check it matches your original if you had
different behavior in mind.
"""

from __future__ import annotations


def get_float_input(
    prompt: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    """Prompt until the user enters a float within [minimum, maximum]."""
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be <= {maximum}.")
            continue
        return value


def get_int_input(
    prompt: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    """Prompt until the user enters an int within [minimum, maximum]."""
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid whole number.")
            continue
        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be <= {maximum}.")
            continue
        return value


def get_choice(prompt: str, choices: list[str]) -> str:
    """Prompt until the user enters one of `choices` (exact match)."""
    while True:
        raw = input(prompt).strip()
        if raw in choices:
            return raw
        print(f"Please enter one of: {', '.join(choices)}.")


def validate_filename(filename: str) -> str | None:
    """Return a sanitized .wav filename, or None if invalid/empty.

    Prints a message and returns None on invalid input rather than raising,
    since main.py treats a None return as "skip saving".
    """
    filename = filename.strip()
    if not filename:
        print("Filename cannot be empty.")
        return None
    if any(c in filename for c in '<>:"/\\|?*'):
        print("Filename contains invalid characters.")
        return None
    if not filename.lower().endswith(".wav"):
        filename += ".wav"
    return filename


if __name__ == "__main__":
    print("Testing utils.py (non-interactive checks only)...\n")

    assert validate_filename("test") == "test.wav"
    assert validate_filename("test.wav") == "test.wav"
    assert validate_filename("  spaced  ") == "spaced.wav"
    assert validate_filename("") is None
    assert validate_filename("bad/name.wav") is None
    assert validate_filename('bad"name') is None

    print("All non-interactive checks passed. utils.py is working correctly!")
    print("(get_float_input / get_int_input / get_choice require manual")
    print(" interactive testing since they read from stdin.)")