import pytest

from examples.dns_level_construction_validation import build, production
from tests.ns.test_dns_levels import check_production


@pytest.mark.parametrize("seed,start", [(101, -1), (202, 1), (303, -1)])
def test_constructed_double_well_ladder_preserves_backtracking(seed, start):
    levels, metadata = build("double_well", seed)
    assert metadata["status"] == "max_levels"
    assert metadata["likelihood_evaluations"] > 0
    result = production("double_well", seed+1000, levels, sweeps=6000, start_sign=start)
    check_production(result)
    assert min(result["switches"]) > 20
    assert min(result["assigned_high_switches"]) > 20
    for witness in result["mechanism_witnesses"]:
        assert witness["minimum_level"] <= 1
        assert witness["crossing_generation_level"] <= 1
        assert witness["witness_levels"][0] >= 3
        assert witness["witness_levels"][-1] >= 3
    assert result["control"]["switches"] == 0
    assert abs(result["control"]["cost_ratio"]-1) < 0.002
    assert all(value is not None for value in result["first_passage_evaluations"])
    assert max(map(int, result["backtracking_depth_counts"])) <= 1
    for row in result["levels"]:
        modal = row["modal_fraction"]
        assert abs(modal["estimate"]-0.5) < 6*modal["se"]+0.004
    # Independently calibrated compression, rather than the selection fraction.
    for row in metadata["rows"]:
        cal = row["calibration"]
        assert abs(cal["ratio"]-row["true_compression"]) < 6*cal["standard_error"]+0.006
