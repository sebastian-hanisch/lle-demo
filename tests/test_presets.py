"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert)."""

import pytest

import lle_constants as C
from lle_evaluation import analyse, make_dataset, verdict


def _measure(p):
    dataset = make_dataset(p["n_tours"], p["q"], p["curvature"], p["noise"], p["seed"])
    a = analyse(dataset, p["k"], p["reg"])
    code = verdict(a, dataset, p["k"], p["reg"])[1]
    return {"verdict": code, "r2_lle": a.r2["lle"], "r2_iso": a.r2["isomap"], "r2_pca": a.r2["pca"], "fid_lle": a.fidelity["lle"], "fid_iso": a.fidelity["isomap"]}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_TOURS_MIN <= p["n_tours"] <= C.N_TOURS_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX
        assert C.CURVATURE_MIN <= p["curvature"] <= C.CURVATURE_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX and C.K_MIN <= p["k"] <= C.K_MAX
        assert p["reg"] in C.REG_CHOICES


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
