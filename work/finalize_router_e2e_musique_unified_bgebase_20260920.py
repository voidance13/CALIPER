#!/usr/bin/env python3
"""Finalize only the frozen MuSiQue 2,403-question E2E rerun."""
import importlib.util
from pathlib import Path

SRC = Path("/mnt/data/caliper/code/CALIPER/work/finalize_router_e2e_20260918.py")
spec = importlib.util.spec_from_file_location("finalize_router_e2e_frozen", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.RUN = mod.BASE / "router_e2e_runs/text_grouped_20260920_unified_bgebase"
mod.DATASETS = ("musique",)
mod.EXPECTED = {"musique": 2403}

if __name__ == "__main__":
    mod.main()
