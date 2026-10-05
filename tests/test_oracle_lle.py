"""Orakel-Tests (unabhängiger Rechenweg): Rekonstruktionsgewichte über das KKT-System statt G⁻¹·1, Nachbarn/Komponenten per sklearn/networkx, Eigenwerte,
Einbettung und Out-of-sample (`transform`) gegen sklearn, Trustworthiness gegen sklearn, R² per QR statt `lstsq`, Out-of-sample-R² von Hand."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

import lle_algorithm as A
import lle_evaluation as E
import lle_scenario as S
from lle_isomap import standardize

nx = pytest.importorskip("networkx")
manifold = pytest.importorskip("sklearn.manifold")
neighbors = pytest.importorskip("sklearn.neighbors")


def _data(rng, kind):
    n, d = int(rng.integers(20, 70)), int(rng.integers(3, 8))
    if kind == 0:
        return rng.normal(size=(n, d)) * rng.uniform(0.3, 4, size=d)
    if kind == 1:
        return S.generate_dataset(n, int(rng.integers(1, 4)), float(rng.uniform(0, 1)), float(rng.uniform(0.1, 1)), 0, int(rng.integers(10 ** 6))).X
    return np.vstack([rng.normal(size=(n // 2, d)), rng.normal(size=(n - n // 2, d)) + 40])           # zwei getrennte Wolken


def test_neighbours_weights_and_components_equal_independent_computations():
    rng = np.random.default_rng(1)
    for t in range(36):
        X = _data(rng, t % 3)
        k, reg = int(rng.integers(2, 12)), float(10 ** rng.uniform(-6, -1))
        Z = standardize(X)
        D = cdist(Z, Z)
        nbrs = A.nearest_neighbours(D, k)
        ref = neighbors.NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
        for i in range(len(Z)):
            assert np.allclose(np.sort(D[i, nbrs[i]]), np.sort(D[i, ref[i]]), atol=1e-9)
        W, errors, _ = A.reconstruction_weights(Z, nbrs, reg)
        for i in range(len(Z)):
            N = Z[nbrs[i]] - Z[i]
            G = N @ N.T + reg * np.trace(N @ N.T) * np.eye(k)
            kkt = np.block([[2 * G, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])       # Lagrange: min wᵀGw  s.t.  Σw = 1
            w = np.linalg.solve(kkt, np.r_[np.zeros(k), 1.0])[:k]
            assert np.allclose(W[i, nbrs[i]], w, atol=1e-6 * max(1.0, np.abs(w).max()))
            assert errors[i] == pytest.approx(float(((W[i] @ Z - Z[i]) ** 2).sum()), rel=1e-9, abs=1e-12)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(Z)))
        graph.add_edges_from((i, int(j)) for i in range(len(Z)) for j in nbrs[i])
        comps = sorted(nx.connected_components(graph), key=len, reverse=True)
        labels = A.component_labels(nbrs)
        assert int(labels.max()) + 1 == len(comps)
        if len(comps) == 1 or len(comps[0]) > len(comps[1]):
            assert set(np.nonzero(labels == 0)[0].tolist()) == set(comps[0])


def test_eigenvalues_embedding_and_out_of_sample_equal_sklearn_on_connected_graphs():
    rng = np.random.default_rng(2)
    checked = 0
    for t in range(40):
        X = _data(rng, t % 2)                                                    # zusammenhängende Wolken
        k, reg = int(rng.integers(4, 14)), float(10 ** rng.uniform(-4, -1))
        m = A.fit_lle(X, k, 2, reg)
        Z = standardize(X)
        sk = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2, reg=reg, eigen_solver="dense").fit(Z)
        assert sk.reconstruction_error_ == pytest.approx(m.eigenvalues[1:3].sum(), rel=1e-6, abs=1e-9)
        ev = m.eigenvalues
        if not (m.connected and ev[1] - ev[0] > 1e-9 and ev[2] - ev[1] > 1e-6 and ev[3] - ev[2] > 1e-6):
            continue                                                             # entartete Eigenwerte: Eigenvektoren nicht eindeutig
        checked += 1
        X_new = X[:8] + 0.05 * rng.normal(size=(8, X.shape[1]))
        y, yk = A.embed_new(m, X_new), sk.transform((X_new - m.mean) / m.scale)
        for c in range(2):
            a, b = m.embedding[:, c], sk.embedding_[:, c] * np.sqrt(len(Z))
            sign = 1.0 if np.abs(a - b).max() <= np.abs(a + b).max() else -1.0
            assert np.abs(a - sign * b).max() < 1e-4
            assert np.allclose(y[:, c], sign * yk[:, c] * np.sqrt(len(Z)), atol=1e-4 * max(1.0, np.abs(y[:, c]).max()))
    assert checked >= 15


def test_trustworthiness_fidelity_r2_and_out_of_sample_equal_by_hand_computations():
    rng = np.random.default_rng(3)
    for _ in range(25):
        ds = S.generate_dataset(int(rng.integers(40, 90)), int(rng.integers(1, 4)), float(rng.uniform(0, 1)), float(rng.uniform(0.1, 1)), 0, int(rng.integers(10 ** 6)))
        m = A.fit_lle(ds.X, 12, 2, 1e-2)
        e2, Z = m.embedding[:, :2], standardize(ds.X)
        k = int(rng.integers(3, 9))
        assert E.trustworthiness(Z, e2, k) == pytest.approx(manifold.trustworthiness(Z, e2, n_neighbors=k), abs=1e-9)
        assert E.distance_fidelity(e2, ds.z) == pytest.approx(np.corrcoef(pdist(e2), pdist(ds.z))[0, 1], abs=1e-9)
        feats = np.column_stack([e2[:, 0], e2[:, 1], e2[:, 0] ** 2, e2[:, 0] * e2[:, 1], e2[:, 1] ** 2, np.ones(len(e2))])
        q, _ = np.linalg.qr(feats)
        r2 = 1 - ((ds.z - q @ (q.T @ ds.z)) ** 2).sum() / ((ds.z - ds.z.mean(0)) ** 2).sum()
        assert E.r2_quadratic(e2, ds.z) == pytest.approx(r2, abs=1e-6)
    ds = S.generate_dataset(120, 2, 1.0, 0.25, 0, 11)
    out = E.out_of_sample(ds, 10, 1e-2)
    train, test = np.arange(96), np.arange(96, 120)
    sk = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2, reg=1e-2, eigen_solver="dense").fit(standardize(ds.X[train]))
    mean, scale = ds.X[train].mean(0), ds.X[train].std(0, ddof=1)
    y = sk.transform((ds.X[test] - mean) / scale) * np.sqrt(96)
    assert np.array_equal(out["train"], train) and np.array_equal(out["test"], test)
    yo = out["y_test"]
    sign = [1.0 if np.abs(yo[:, c] - y[:, c]).max() <= np.abs(yo[:, c] + y[:, c]).max() else -1.0 for c in range(2)]
    assert np.allclose(yo, y * sign, atol=1e-3)
    quad = lambda e: np.column_stack([e[:, 0], e[:, 1], e[:, 0] ** 2, e[:, 0] * e[:, 1], e[:, 1] ** 2, np.ones(len(e))])
    beta = np.linalg.pinv(quad(out["model"].embedding)) @ ds.z[train]
    resid = ds.z[test] - quad(yo) @ beta
    assert out["r2_test"] == pytest.approx(1 - (resid ** 2).sum() / ((ds.z[test] - ds.z[test].mean(0)) ** 2).sum(), abs=1e-6)
