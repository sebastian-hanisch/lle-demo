import numpy as np
import pytest
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import LocallyLinearEmbedding

from lle_algorithm import (
    SingularNeighbourhood, component_labels, embed_new, fit_lle, nearest_neighbours, reconstruction_weights,
)
from lle_isomap import fit_isomap, pairwise_distances, standardize
from lle_scenario import generate_dataset


def _data(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 3, n)
    X = np.column_stack([np.cos(t), np.sin(t), t, 0.1 * rng.standard_normal(n), 0.1 * rng.standard_normal(n), 0.1 * rng.standard_normal(n)])[:, :d]
    return X


def _weights(X, k, reg):
    Z = standardize(X)
    nbrs = nearest_neighbours(pairwise_distances(Z), k)
    return Z, nbrs, *reconstruction_weights(Z, nbrs, reg)


def test_weights_sum_to_one_and_are_supported_on_the_neighbours():
    Z, nbrs, W, errors, _ = _weights(_data(), 8, 1e-3)
    assert np.allclose(W.sum(1), 1.0)
    for i in range(len(Z)):
        assert set(np.nonzero(W[i])[0]) <= set(nbrs[i])


def test_exact_reconstruction_for_an_affine_neighbourhood():
    """Liegt eine Tour exakt in der affinen Hülle ihrer Nachbarn, ist der Rekonstruktionsfehler ~0 (die lokale Gram-Matrix ist dann singulär, ihr Kern liefert die Gewichte -
    deshalb eine winzige Regularisierung statt 0)."""
    rng = np.random.default_rng(1)
    nbrs = rng.standard_normal((4, 6))
    w_true = np.array([0.4, 0.3, 0.2, 0.1])
    point = w_true @ nbrs
    Z = np.vstack([point, nbrs])
    W, errors, _ = reconstruction_weights(Z, np.array([[1, 2, 3, 4]] + [[0, 1, 2, 3]] * 4), 1e-12)
    assert errors[0] < 1e-10 and np.allclose(W[0, 1:], w_true, atol=1e-5)


def test_weights_match_a_hand_solution_for_two_neighbours():
    # Tour bei (0,0); Nachbarn (1,0) und (0,1): affine Kombination mit Summe 1 ist ein Punkt der Verbindungsgeraden x+y=1 - der nächste zum Ursprung (0.5, 0.5) -> w = (0.5, 0.5)
    Z = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    W, errors, _ = reconstruction_weights(Z, np.array([[1, 2], [0, 2], [0, 1]]), 0.0)
    assert np.allclose(W[0, 1:], [0.5, 0.5]) and abs(errors[0] - 0.5) < 1e-12


def test_m_matrix_is_positive_semidefinite_with_a_zero_eigenvalue_for_the_constant_vector():
    X = _data()
    Z, nbrs, W, _, _ = _weights(X, 8, 1e-3)
    A = np.eye(len(Z)) - W
    M = A.T @ A
    assert np.linalg.eigvalsh((M + M.T) / 2).min() > -1e-10
    assert np.allclose(M @ np.ones(len(Z)), 0.0, atol=1e-10)
    model = fit_lle(X, 8, 2, 1e-3)
    assert abs(model.eigenvalues[0]) < 1e-10 and model.eigenvalues[1] > 1e-8


def test_embedding_is_centred_with_unit_variance_and_matches_sklearn():
    ds = generate_dataset(200, 2, 1.0, 0.25, 0, 7)
    model = fit_lle(ds.X, 14, 2, 1e-2)
    assert np.allclose(model.embedding.mean(0), 0, atol=1e-9) and np.allclose((model.embedding ** 2).mean(0), 1.0, atol=1e-9)
    # sklearn (gleiche Standardisierung vorab, gleiche reg-Konvention): Eigenwerte gleich, Einbettung bis auf Vorzeichen gleich
    Z = standardize(ds.X)
    sk = LocallyLinearEmbedding(n_neighbors=14, n_components=2, reg=1e-2, eigen_solver="dense").fit(Z)
    assert abs(sk.reconstruction_error_ - model.eigenvalues[1:3].sum()) < 1e-6
    for c in range(2):
        a, b = model.embedding[:, c], sk.embedding_[:, c] * np.sqrt(len(Z))
        assert min(np.abs(a - b).max(), np.abs(a + b).max()) < 1e-5


def test_k_above_dimension_without_regularisation_is_reported_not_repaired():
    X = _data(d=6)
    with pytest.raises(SingularNeighbourhood):
        fit_lle(X, 9, 2, 0.0)
    fit_lle(X, 9, 2, 1e-3)                        # mit Regularisierung definiert
    fit_lle(X, 5, 2, 0.0)                         # k <= Dimension: auch ohne definiert


def test_disconnected_graph_is_detected_and_has_multiple_zero_eigenvalues():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.standard_normal((40, 4)), rng.standard_normal((40, 4)) + 30.0])
    model = fit_lle(X, 6, 2, 1e-3)
    assert not model.connected and int(model.components.max()) == 1
    assert (model.eigenvalues[:2] < 1e-9).all()


def test_component_labels_hand_graph():
    nbrs = np.array([[1], [0], [3], [2], [3]])
    labels = component_labels(nbrs)
    assert labels[0] == labels[1] and labels[2] == labels[3] == labels[4] and labels[0] != labels[2]
    assert labels[2] == 0                          # größte Komponente hat Label 0


def test_out_of_sample_reproduces_smooth_coordinates_of_new_points():
    ds = generate_dataset(300, 2, 1.0, 0.1, 0, 5)
    train, test = np.arange(240), np.arange(240, 300)
    model = fit_lle(ds.X[train], 14, 2, 1e-2)
    y = embed_new(model, ds.X[test])
    assert y.shape == (60, 2)
    # neue Punkte liegen im Bereich der Trainings-Einbettung und ihre nächsten Trainings-Nachbarn liegen im Faktorraum nahe
    assert (np.abs(y) <= np.abs(model.embedding).max(0) * 1.5).all()


def test_out_of_sample_of_a_training_point_lands_near_its_own_coordinates():
    X = _data(80)
    model = fit_lle(X, 5, 2, 1e-3)
    # ein Trainingspunkt, mit leicht verschobenem Nachbar-Set (Selbst-Abstand 0 -> er ist sein eigener nächster Nachbar): Einbettung liegt nahe seiner eigenen
    y = embed_new(model, X[:10])
    assert np.abs(y - model.embedding[:10]).max() < 0.2 * np.abs(model.embedding).max()


def test_isomap_copy_matches_the_isomap_demo_reference_values():
    ds = generate_dataset(300, 2, 1.0, 0.25, 0, 7)
    r = fit_isomap(ds.X, 8, 2)
    assert abs(float(np.abs(r.embedding).sum()) - 1786.6726227517283) < 1e-6 and abs(float(r.geodesic.sum()) - 621595.593373937) < 1e-3


def test_isomap_copy_geodesics_match_scipy_dijkstra():
    ds = generate_dataset(120, 2, 1.0, 0.25, 0, 3)
    r = fit_isomap(ds.X, 8, 2)
    w = np.where(np.isfinite(r.weights) & (r.weights > 0), r.weights, 0.0)
    sp = shortest_path(w[np.ix_(r.indices, r.indices)], directed=False)
    assert np.allclose(sp, r.geodesic, atol=1e-8)
