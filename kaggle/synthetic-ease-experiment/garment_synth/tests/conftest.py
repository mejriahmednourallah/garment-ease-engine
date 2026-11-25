import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_synthetic_garments as gen  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return gen.GenConfig(seed=777, **gen.PRESETS["small"])


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, cfg):
    out = tmp_path_factory.mktemp("synthetic")
    gen.write_all(cfg, out)
    return out


@pytest.fixture(scope="session")
def sets(data_dir):
    import pandas as pd
    res = {}
    for kind in ("main", "stress"):
        res[kind] = {p.stem: pd.read_csv(p) for p in (data_dir / kind).glob("*.csv")}
    return res
