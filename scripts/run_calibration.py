"""Compatibility entry point for repository-disjoint pcg.validation.

Run: python scripts/run_calibration.py --curated --execution local
"""

import sys
import os

# Ensure project root is on the path so `pcg` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcg.validation import main

if __name__ == "__main__":
    main()
