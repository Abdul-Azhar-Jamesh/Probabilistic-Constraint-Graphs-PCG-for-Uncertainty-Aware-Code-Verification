"""End-to-end calibration runner.

Generates the mutation corpus, builds the training set, fits models,
validates on the holdout case, and saves all results.

Run:  python scripts/run_calibration.py
"""

import sys
import os

# Ensure project root is on the path so `pcg` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcg.fit_weights import main

if __name__ == "__main__":
    main()
