#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

_PYTORCH_CPU_REQUIREMENT = re.compile(
    r"^(torch(?:vision)?)==(\d+(?:\.\d+)+)\+cpu(?=\s|$)",
    re.MULTILINE,
)


def prepare_audit_lock(lock_text: str) -> str:
    # Advisory databases identify CPU wheels by their canonical public version.
    # Keep every wheel hash intact and normalize only the two PyTorch headers.
    return _PYTORCH_CPU_REQUIREMENT.sub(r"\1==\2", lock_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a hash-locked PyTorch CPU requirements report for pip-audit."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("input and output paths must differ")

    lock_text = args.input.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prepare_audit_lock(lock_text), encoding="utf-8")


if __name__ == "__main__":
    main()
