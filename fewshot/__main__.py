"""The ``fewshot`` command (and ``python -m fewshot``): the CLI, with Ctrl-C while its modules are
still loading ending as quietly as it does later on."""

import importlib
import sys


def main() -> None:
    """Load the CLI and run it; Ctrl-C meanwhile prints "Interrupted" and exits with status 130."""
    try:
        cli = importlib.import_module("fewshot.cli")
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted\n")
        sys.exit(130)
    cli.main()


if __name__ == "__main__":
    main()
