"""Experiments run from the command line and leave auditable, machine-readable output."""
import json

import pytest

from experiments.run import REGISTRY, main


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_experiment_quick_run_writes_results_and_provenance(name, tmp_path):
    if name in ("benchmark_engine", "benchmark_complexity"):
        from engine import cpp_engine
        if not cpp_engine.available():
            pytest.skip("C++ extension not built")
    assert main([name, "--seed", "7", "--quick", "--out", str(tmp_path)]) == 0
    out = tmp_path / name / "quick"
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["experiment"] == name and prov["seed"] == 7 and prov["quick_mode"] is True
    assert prov["git"]["commit"] is not None and "dirty" in prov["git"]
    assert prov["parameters"] and prov["dataset"]
    assert prov["result_file_sha256"] and all(len(h) == 64 for h in prov["result_file_sha256"].values())
    assert any(out.glob("*.csv")) and any(out.glob("*.png"))


def test_same_seed_same_results_bytes(tmp_path):
    for i in (1, 2):
        main(["as_assumptions", "--seed", "3", "--quick", "--out", str(tmp_path / str(i))])
    a = (tmp_path / "1/as_assumptions/quick/intensity.csv").read_bytes()
    b = (tmp_path / "2/as_assumptions/quick/intensity.csv").read_bytes()
    assert a == b


def test_unknown_experiment_and_list(capsys):
    assert main(["nope"]) == 2
    assert main(["--list"]) == 0
    assert "inventory" in capsys.readouterr().out
