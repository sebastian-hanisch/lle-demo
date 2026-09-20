import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness

import lle_constants as C
from lle_evaluation import (
    analyse, distance_fidelity, k_sweep, make_dataset, out_of_sample, pca_project, r2_quadratic, reg_sweep, timing_sweep, trustworthiness, verdict,
)
from lle_scenario import generate_dataset


def test_scenario_is_bit_identical_to_the_pca_demo_generator():
    """Eingefrorene Referenzwerte aus pca-demo (`generate_dataset`, gleiche Argumente): die Kopie darf nicht abweichen."""
    d = generate_dataset(300, 2, 1.0, 0.25, 0, 7)
    assert d.X.shape == (300, 12) and abs(float(d.X.sum()) - 14553337.310875032) < 1e-6
    assert np.allclose(d.X[0, :3], [54469.13732407575, 74.83224937121233, 767.2459701758783]) and abs(float(d.z.sum()) - (-80.2378453142044)) < 1e-9
    assert abs(float(generate_dataset(200, 3, 0.4, 0.3, 5, 42).X.sum()) - 10763498.969287368) < 1e-6


def test_make_dataset_is_deterministic_and_seed_dependent():
    a, b = make_dataset(150, 2, 0.5, 0.2, 3), make_dataset(150, 2, 0.5, 0.2, 3)
    assert np.array_equal(a.X, b.X) and not np.array_equal(a.X, make_dataset(150, 2, 0.5, 0.2, 4).X)


def test_trustworthiness_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    for embedding in (X[:, :2], rng.standard_normal((80, 2)), pca_project(X)):
        assert abs(trustworthiness(X, embedding, 8) - sk_trustworthiness(X, embedding, n_neighbors=8)) < 1e-9


def test_r2_quadratic_recovers_monotone_reparametrisations_and_rejects_noise():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((200, 2))
    coords = np.column_stack([z[:, 0] + 0.3 * z[:, 0] ** 2, z[:, 1] + 0.2 * z[:, 1] ** 2])
    assert r2_quadratic(coords, z) > 0.9
    assert r2_quadratic(rng.standard_normal((200, 2)), z) < 0.1


def test_distance_fidelity_is_one_for_a_scaled_copy_and_low_for_noise():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((100, 2))
    assert distance_fidelity(3.0 * z, z) > 0.999999
    assert distance_fidelity(rng.standard_normal((100, 2)), z) < 0.2


def test_analysis_lle_and_isomap_beat_pca_on_the_default_surface_and_lle_costs_fidelity():
    ds = make_dataset(300, 2, 1.0, 0.25, 7)
    a = analyse(ds, 14, 1e-2)
    assert a.r2["lle"] > 0.9 and a.r2["isomap"] > 0.9 and a.r2["pca"] < 0.6
    assert a.fidelity["lle"] < a.fidelity["isomap"]                     # Kontrast: LLE erhält Nachbarschaften, keine Abstände
    assert a.oos is not None and a.oos["r2_test"] > 0.3


def test_analysis_reports_singular_neighbourhoods_and_keeps_isomap_and_pca():
    ds = make_dataset(300, 2, 1.0, 0.25, 7)
    a = analyse(ds, 20, 0.0)
    assert a.lle is None and a.oos is None and a.r2["lle"] is None and a.r2["isomap"] > 0.9
    level, code, data = verdict(a, ds, 20, 0.0)
    assert (level, code) == ("error", "undefined") and data["k"] == 20


def test_out_of_sample_holds_out_the_last_fraction_and_is_none_when_singular():
    ds = make_dataset(300, 2, 1.0, 0.25, 7)
    oos = out_of_sample(ds, 14, 1e-2)
    assert len(oos["test"]) == 60 and oos["test"][0] == 240 and oos["y_test"].shape == (60, 2)
    assert out_of_sample(ds, 20, 0.0) is None


def test_sweeps_are_deterministic_and_show_the_windows():
    a, b = k_sweep(2, 1.0, 0.25, 1e-2, ks=(3, 14)), k_sweep(2, 1.0, 0.25, 1e-2, ks=(3, 14))
    assert a == b
    assert a[0]["r2_lle"] < 0.6 < 0.9 < a[1]["r2_lle"]                  # k = 3 zu klein, k = 14 gut
    r = reg_sweep(2, 1.0, 0.25, 14, regs=(1e-3, 1e-2))
    assert r[0]["r2_lle"] < r[1]["r2_lle"]                                 # zu schwache Regularisierung schadet bei k > d
    assert reg_sweep(2, 1.0, 0.25, 20, regs=(0.0,))[0]["r2_lle"] is None


def test_sweep_seeds_are_separate_from_demo_seeds():
    assert min(C.SWEEP_SEEDS) >= 100_000 > C.DEFAULT_SEED


@pytest.mark.parametrize("q,curv,noise,k,reg,code", [
    (2, 1.0, 0.25, 14, 1e-2, "lle_wins"),
    (2, 1.0, 0.25, 14, 1e-3, "reg_weak"),
    (2, 1.0, 0.25, 3, 1e-2, "k_small"),
    (2, 1.0, 0.8, 14, 1e-2, "noise"),
    (2, 0.0, 0.25, 14, 1e-2, "no_advantage"),
])
def test_verdict_codes(q, curv, noise, k, reg, code):
    ds = make_dataset(300, q, curv, noise, 7)
    assert verdict(analyse(ds, k, reg), ds, k, reg)[1] == code


def test_timing_sweep_has_the_expected_shape_and_lle_beats_isomap_at_larger_n():
    rows = timing_sweep(ns=(100, 600))
    assert [r["n"] for r in rows] == [100, 600] and all(r[key] > 0 for r in rows for key in ("lle", "isomap", "pca"))
    assert rows[1]["lle"] < rows[1]["isomap"]                            # bei wenigen Touren ist Isomap schneller, ab einigen hundert LLE (gemessen: ~4x bei n = 600)
