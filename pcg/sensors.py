"""Versioned observation likelihoods; defaults are assumptions, not calibration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SENSORS = {
    "compile": {"flag_given_defect": 0.40, "flag_given_clean": 0.001},
    "static": {"flag_given_defect": 0.65, "flag_given_clean": 0.15},
    "exec": {"flag_given_defect": 0.95, "flag_given_clean": 0.01},
    "llm": {"flag_given_defect": 0.55, "flag_given_clean": 0.35},
    "critic": {"flag_given_defect": 0.65, "flag_given_clean": 0.25},
}

# Assumed sensitivity of partial oracles, not learned calibration values.
ORACLE_SENSITIVITY_SCALE = {"annotation": 0.25, "property": 0.5, "metamorphic": 0.5}


def test_sensitivity(model: "SensorModel", scope: str) -> float:
    return model.likelihoods["exec"][
        "flag_given_defect"
    ] * ORACLE_SENSITIVITY_SCALE.get(scope, 1.0)


@dataclass(frozen=True)
class SensorModel:
    likelihoods: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            source: dict(values) for source, values in DEFAULT_SENSORS.items()
        }
    )
    prior_coefficients: dict[str, float] | None = None
    calibrated: bool = False
    posterior_threshold: float = 0.5
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            type(self.posterior_threshold) not in {float, int}
            or not 0 < self.posterior_threshold < 1
        ):
            raise ValueError("posterior threshold must be in (0, 1)")
        if type(self.calibrated) is not bool or not isinstance(self.provenance, dict):
            raise ValueError("invalid model calibration metadata")
        if not isinstance(self.likelihoods, dict) or set(self.likelihoods) != set(
            DEFAULT_SENSORS
        ):
            raise ValueError("model must provide all supported sensor likelihoods")
        for source, probabilities in self.likelihoods.items():
            if not isinstance(probabilities, dict) or set(probabilities) != {
                "flag_given_defect",
                "flag_given_clean",
            }:
                raise ValueError(f"invalid likelihood keys for {source}")
            if any(
                type(p) not in {float, int} or not math.isfinite(p) or not 0 < p < 1
                for p in probabilities.values()
            ):
                raise ValueError(f"invalid observation probabilities for {source}")
        if self.prior_coefficients is not None:
            expected = {"intercept", "log_cyclomatic", "log_loc", "depth", "n_params"}
            if (
                not isinstance(self.prior_coefficients, dict)
                or set(self.prior_coefficients) != expected
                or any(
                    type(v) not in {float, int} or not math.isfinite(v)
                    for v in self.prior_coefficients.values()
                )
            ):
                raise ValueError("invalid logistic prior coefficients")

    @classmethod
    def load(cls, path: str | Path) -> "SensorModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("model") != "latent-defect-noisy-or-v1"
        ):
            raise ValueError(
                "model file is not compatible with the deployed inference procedure"
            )
        return cls(
            likelihoods=payload.get("likelihoods", {}),
            prior_coefficients=payload.get("prior_coefficients"),
            calibrated=payload.get("calibrated", False),
            posterior_threshold=payload.get("posterior_threshold", 0.5),
            provenance=payload.get("provenance", {}),
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return {
            "schema_version": 1,
            "model": "latent-defect-noisy-or-v1",
            **asdict(self),
        }
