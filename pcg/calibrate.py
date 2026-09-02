"""Calibrate SOURCE_RELIABILITY and prior coefficients from labelled data.

This module is a thin compatibility shim: the canonical implementation lives
in :mod:`pcg.fit_weights` (used by ``scripts/run_calibration.py``). Both
``python -m pcg.calibrate`` and ``python -m pcg.fit_weights`` therefore run
the same code.

Fits logistic regression models on the mutation-based training set to replace
hand-tuned weights in inference.py.  Two models are trained separately:

  1. **Prior model**: structural features only → P(correct) before evidence.
  2. **Evidence model**: evidence counts/weights → P(correct | evidence).

A joint model on all features is also fitted for comparison.
"""

from .fit_weights import *  # noqa: F401,F403
from .fit_weights import main

if __name__ == "__main__":
    main()
