"""PyInstaller entry point for the InkedIn CLI/UI executable.

The packaged exe covers the torch-free core: all import formats, theme modes,
the web UI, export. ML modes (fast/ai/translate) tell the user to run from a
Python environment — bundling torch would produce a multi-GB binary.
"""

import multiprocessing
import sys

from inkedin_core.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # required in frozen Windows binaries
    sys.exit(main())
