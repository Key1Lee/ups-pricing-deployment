"""One storage root makes local runs, isolated tests, and Docker use the same code."""

import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42
HISTORY_SEED = 2026  # Final independent snapshot; development used seed 42.
DEMO_DAY = "2025-08-29"
SERVICES = ("ground", "two_day", "next_day")
SEGMENTS = ("occasional", "small_business", "enterprise")


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data(self):
        return self.root / "data"

    @property
    def artifacts(self):
        return self.root / "artifacts"

    @property
    def db(self):
        return self.data / "shipping.sqlite"

    def ensure(self):
        self.data.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        return self


def paths():
    default = Path(__file__).resolve().parents[2]
    return Paths(Path(os.environ.get("UPS_PRICING_HOME", default)).resolve()).ensure()
