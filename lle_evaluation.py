"""Auswertung: findet LLE die wahren Faktoren zurück - und wo bricht es? Alle Kennzahlen werden am Datensatz gemessen; die wahren latenten Faktoren z
sind bekannt (Lieferrouten-Erzeugung). LLE, Isomap (Vergleichspartner) und PCA werden mit denselben Messungen bewertet.

- **R² der wahren Faktoren**: Rekonstruktion von z aus den ersten zwei Koordinaten per quadratischer Regression (monotone Umparametrisierungen werden nicht bestraft).
- **Abstandstreue**: Pearson-Korrelation der Paarabstände in der 2-D-Einbettung mit den Paarabständen der wahren Faktoren - LLE erhält Nachbarschaften, keine Abstände.
- **Trustworthiness** (Venna & Kaski): bleiben Nachbarn Nachbarn?
- **Out-of-sample**: 20 % der Touren zurückhalten, LLE auf dem Rest trainieren, die zurückgehaltenen über ihre Rekonstruktionsgewichte einbetten; R² der wahren Faktoren
  über eine auf den Trainingstouren angepasste quadratische Abbildung Einbettung -> Faktoren."""

import time
from dataclasses import dataclass

import numpy as np

import lle_constants as C
from lle_algorithm import SingularNeighbourhood, embed_new, fit_lle
from lle_isomap import fit_isomap, pairwise_distances, standardize
from lle_scenario import generate_dataset


def trustworthiness(X_high, X_low, n_neighbors=C.TRUST_NEIGHBORS):
    """Trustworthiness (Venna & Kaski, 2001): Anteil der Nachbarn im Einbettungsraum, die auch im Originalraum echte Nachbarn sind, mit
    Rang-Strafe für eingeschleppte Fremde. 1 = perfekt. Eigene Implementierung, gegen sklearn geprüft (nur im Test)."""
    n = len(X_high)
    k = n_neighbors
    d_high = np.linalg.norm(X_high[:, None, :] - X_high[None, :, :], axis=-1)
    d_low = np.linalg.norm(X_low[:, None, :] - X_low[None, :, :], axis=-1)
    np.fill_diagonal(d_high, np.inf)
    np.fill_diagonal(d_low, np.inf)
    ranks_high = np.argsort(np.argsort(d_high, axis=1), axis=1) + 1            # Rang 1 = nächster Nachbar
    neighbors_low = np.argsort(d_low, axis=1)[:, :k]
    penalty = 0.0
    for i in range(n):
        r = ranks_high[i, neighbors_low[i]]
        penalty += float(np.maximum(r - k, 0).sum())
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty


def r2_quadratic(coords2, z):
    """R² der Rekonstruktion von z aus zwei Koordinaten (quadratische Regression, Mittel über die Faktoren, gewichtet mit ihrer Varianz)."""
    e1, e2 = coords2[:, 0], coords2[:, 1]
    A = np.column_stack([e1, e2, e1 ** 2, e1 * e2, e2 ** 2, np.ones(len(e1))])
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(1.0 - (z - A @ beta).var(0).sum() / z.var(0).sum())


def pca_project(X, n_components=2):
    Z = standardize(X)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:n_components].T


def distance_fidelity(coords2, z):
    iu = np.triu_indices(len(z), 1)
    return float(np.corrcoef(pairwise_distances(coords2)[iu], pairwise_distances(z)[iu])[0, 1])


def _quad_features(e):
    return np.column_stack([e[:, 0], e[:, 1], e[:, 0] ** 2, e[:, 0] * e[:, 1], e[:, 1] ** 2, np.ones(len(e))])


def out_of_sample(dataset, k, reg, fraction=C.HOLDOUT_FRACTION):
    """Zurückgehaltene Touren (die letzten `fraction`) über die Rekonstruktionsgewichte einbetten. -> dict oder None (singulär)."""
    n = dataset.n
    n_test = max(10, int(round(fraction * n)))
    train, test = np.arange(n - n_test), np.arange(n - n_test, n)
    try:
        model = fit_lle(dataset.X[train], k, 2, reg)
    except SingularNeighbourhood:
        return None
    y = embed_new(model, dataset.X[test])
    beta, *_ = np.linalg.lstsq(_quad_features(model.embedding), dataset.z[train], rcond=None)
    z_test = dataset.z[test]
    resid = z_test - _quad_features(y) @ beta
    return {"train": train, "test": test, "model": model, "y_test": y, "r2_test": float(1 - resid.var(0).sum() / z_test.var(0).sum()),
            "r2_train": r2_quadratic(model.embedding[:, :2], dataset.z[train])}


@dataclass(frozen=True)
class Analysis:
    lle: object                      # LLEModel oder None (singulär)
    singular_message: str
    isomap: object
    lle_2d: object
    iso_2d: np.ndarray
    pca_2d: np.ndarray
    iso_indices: np.ndarray
    r2: dict                         # {"lle", "isomap", "pca"}
    trust: dict
    fidelity: dict
    oos: object


def analyse(dataset, k, reg):
    Z = standardize(dataset.X)
    pca2 = pca_project(dataset.X)
    iso = fit_isomap(dataset.X, k, 2)
    iso2 = iso.embedding[:, :2]
    try:
        lle, message = fit_lle(dataset.X, k, C.N_COMPONENTS_MAX, reg), ""
    except SingularNeighbourhood as exc:
        lle, message = None, str(exc)
    lle2 = None if lle is None else lle.embedding[:, :2]
    r2 = {"pca": r2_quadratic(pca2, dataset.z), "isomap": r2_quadratic(iso2, dataset.z[iso.indices]),
          "lle": None if lle is None else r2_quadratic(lle2, dataset.z)}
    trust = {"pca": trustworthiness(Z, pca2), "isomap": trustworthiness(Z[iso.indices], iso2), "lle": None if lle is None else trustworthiness(Z, lle2)}
    fid = {"pca": distance_fidelity(pca2, dataset.z), "isomap": distance_fidelity(iso2, dataset.z[iso.indices]),
           "lle": None if lle is None else distance_fidelity(lle2, dataset.z)}
    return Analysis(lle=lle, singular_message=message, isomap=iso, lle_2d=lle2, iso_2d=iso2, pca_2d=pca2, iso_indices=iso.indices, r2=r2, trust=trust,
                    fidelity=fid, oos=None if lle is None else out_of_sample(dataset, k, reg))


def verdict(analysis, dataset, k, reg):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten)."""
    a = analysis
    if a.lle is None:
        return "error", "undefined", {"k": k, "d": C.N_FEATURES, "message": a.singular_message}
    m = a.lle
    data = {"k": k, "d": C.N_FEATURES, "reg": reg, "r2_lle": a.r2["lle"], "r2_iso": a.r2["isomap"], "r2_pca": a.r2["pca"], "trust_lle": a.trust["lle"],
            "trust_iso": a.trust["isomap"], "fid_lle": a.fidelity["lle"], "fid_iso": a.fidelity["isomap"], "components": int(m.components.max()) + 1,
            "singular": m.singular_share, "oos": None if a.oos is None else a.oos["r2_test"]}
    if not m.connected:
        return "warning", "disconnected", data
    if m.singular_share >= 0.5 and reg <= 1e-3:
        return "warning", "reg_weak", data
    if k <= 5 and a.r2["lle"] < 0.6:
        return "warning", "k_small", data
    if dataset.noise >= 0.4 and a.r2["lle"] < 0.6:
        return "warning", "noise", data
    if dataset.curvature == 0 and a.r2["lle"] - a.r2["pca"] < 0.03:
        return "info", "no_advantage", data
    if a.r2["lle"] - a.r2["pca"] >= 0.10:
        return "success", "lle_wins", data
    return "info", "neutral", data


def make_dataset(n_tours, q, curvature, noise, seed):
    return generate_dataset(n_tours, q, curvature, noise, 0, seed)


def _lle_metrics(dataset, k, reg):
    try:
        model = fit_lle(dataset.X, k, 2, reg)
    except SingularNeighbourhood:
        return None
    e = model.embedding[:, :2]
    return r2_quadratic(e, dataset.z), distance_fidelity(e, dataset.z), trustworthiness(standardize(dataset.X), e), model.reconstruction_error


def k_sweep(q, curvature, noise, reg, n_tours=C.SWEEP_N_TOURS, ks=C.SWEEP_KS, seeds=C.SWEEP_SEEDS):
    """Feste Sweep-Seeds: je k mittleres R², Abstandstreue und Trustworthiness von LLE sowie R² von Isomap (Vergleich)."""
    rows = []
    for k in ks:
        r2l, fid, tr, r2i = [], [], [], []
        for seed in seeds:
            ds = make_dataset(n_tours, q, curvature, noise, seed)
            m = _lle_metrics(ds, k, reg)
            if m is None:
                continue
            r2l.append(m[0]), fid.append(m[1]), tr.append(m[2])
            iso = fit_isomap(ds.X, k, 2)
            r2i.append(r2_quadratic(iso.embedding[:, :2], ds.z[iso.indices]))
        rows.append({"k": int(k), "r2_lle": float(np.mean(r2l)) if r2l else None, "fid_lle": float(np.mean(fid)) if fid else None,
                     "trust_lle": float(np.mean(tr)) if tr else None, "r2_iso": float(np.mean(r2i)) if r2i else None})
    return rows


def reg_sweep(q, curvature, noise, k, n_tours=C.SWEEP_N_TOURS, regs=C.SWEEP_REGS, seeds=C.SWEEP_SEEDS):
    """Je Regularisierung mittleres R² und Rekonstruktionsfehler bei festem k."""
    rows = []
    for reg in regs:
        r2l, err = [], []
        for seed in seeds:
            m = _lle_metrics(make_dataset(n_tours, q, curvature, noise, seed), k, reg)
            if m is not None:
                r2l.append(m[0]), err.append(m[3])
        rows.append({"reg": float(reg), "r2_lle": float(np.mean(r2l)) if r2l else None, "error": float(np.mean(err)) if err else None})
    return rows


def timing_sweep(ns=C.TIMING_NS, k=C.TIMING_K, q=2, seed=100_000):
    """Gemessene Rechenzeit (Sekunden) von LLE, Isomap und PCA für wachsende n (eigene Messung, Rechner-abhängig)."""
    rows = []
    for n in ns:
        ds = generate_dataset(n, q, 0.0, 0.1, 0, seed)
        out = {"n": int(n)}
        for name, fn in (("lle", lambda: fit_lle(ds.X, k, 2, 1e-3)), ("isomap", lambda: fit_isomap(ds.X, k, 2)), ("pca", lambda: pca_project(ds.X))):
            t = time.perf_counter()
            fn()
            out[name] = time.perf_counter() - t
        rows.append(out)
    return rows
