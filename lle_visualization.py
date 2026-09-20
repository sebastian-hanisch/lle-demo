"""Plotly-Visualisierungen der LLE-Demo: Nachbarn und Rekonstruktion einer Tour, Gewichts-Balken, Nachbarschaftsgraph, Einbettungen (LLE / Isomap / PCA),
Eigenwertspektrum, Abstandstreue, k- und Regularisierungs-Sweep, Out-of-sample und Rechenzeit. Alle Figuren laufen durch `lock_axes`
(Touch-Scrolling-Konvention des Portfolios: keine Zoom-/Pan-Gesten im Chart)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREEN, RED, GRAY = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _scatter(coords, color, name="Touren", size=7, showscale=False, label="latenter Faktor 1", opacity=1.0):
    return go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", name=name, hoverinfo="skip",
        marker=dict(color=color, colorscale="Viridis", size=size, showscale=showscale, opacity=opacity, line=dict(width=0.5, color="white"),
                    colorbar=dict(title=label) if showscale else None),
    )


def build_view(coords, color, edges=None, focus=None, neighbours=None, weights=None, reconstruction=None):
    """2-D-Ansicht (erste zwei Hauptkomponenten); optional Kanten (Nachbarn je Tour), eine gewählte Tour mit ihren Nachbarn (Größe = |Gewicht|) und ihre Rekonstruktion."""
    fig = go.Figure()
    if edges is not None and len(edges):
        xs, ys = [], []
        for i, j in edges:
            xs += [coords[i, 0], coords[j, 0], None]
            ys += [coords[i, 1], coords[j, 1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(120,120,120,0.25)", width=1), name="Nachbarschaften", hoverinfo="skip"))
    fig.add_trace(_scatter(coords, color, showscale=True, opacity=0.55 if focus is not None else 1.0))
    if focus is not None:
        nb = np.asarray(neighbours)
        sizes = 11 if weights is None else 9 + 30 * np.abs(weights) / max(np.abs(weights).max(), 1e-9)
        fig.add_trace(go.Scatter(x=coords[nb, 0], y=coords[nb, 1], mode="markers", name="Nachbarn" if weights is None else "Nachbarn (Größe = |Gewicht|)", hoverinfo="skip",
                                 marker=dict(color=ORANGE, size=sizes, line=dict(width=1, color="#14233B"))))
        fig.add_trace(go.Scatter(x=[coords[focus, 0]], y=[coords[focus, 1]], mode="markers", name="gewählte Tour", hoverinfo="skip",
                                 marker=dict(color=RED, size=14, symbol="star", line=dict(width=1, color="#14233B"))))
        if reconstruction is not None:
            fig.add_trace(go.Scatter(x=[reconstruction[0]], y=[reconstruction[1]], mode="markers", name="Rekonstruktion aus den Nachbarn", hoverinfo="skip",
                                     marker=dict(color=GREEN, size=12, symbol="diamond-open", line=dict(width=3, color=GREEN))))
    fig.update_xaxes(title="PC1 (Ansicht)")
    fig.update_yaxes(title="PC2 (Ansicht)")
    fig.update_layout(template="plotly_white", height=460, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_weights_bar(weights):
    """Rekonstruktionsgewichte einer Tour (Summe 1); negative Gewichte sind erlaubt (die Tour liegt dann außerhalb der Nachbarn)."""
    colors = [BLUE if w >= 0 else RED for w in weights]
    fig = go.Figure(go.Bar(x=[f"N{i + 1}" for i in range(len(weights))], y=weights, marker_color=colors, hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.update_yaxes(title="Gewicht")
    fig.update_xaxes(title="Nachbarn (nach Abstand geordnet)")
    fig.update_layout(template="plotly_white", height=260, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return lock_axes(fig)


def build_embedding(coords, color, title_x, title_y, label="latenter Faktor 1", height=380):
    fig = go.Figure(_scatter(coords, color, showscale=True, label=label))
    fig.update_xaxes(title=title_x)
    fig.update_yaxes(title=title_y)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=20, b=10))
    return lock_axes(fig)


def build_spectrum(eigenvalues, n_components=2):
    """Die kleinsten Eigenwerte von M = (I − W)^T (I − W): der erste (≈ 0) gehört zum konstanten Vektor und wird übersprungen; die nächsten liefern die Koordinaten."""
    idx = np.arange(len(eigenvalues))
    colors = [GRAY if i == 0 else (ORANGE if i <= n_components else "#9db8d2") for i in idx]
    fig = go.Figure(go.Bar(x=[f"{i + 1}" for i in idx], y=eigenvalues, marker_color=colors, hovertemplate="Eigenwert %{x}: %{y:.5f}<extra></extra>"))
    fig.update_xaxes(title="Eigenwert (aufsteigend)")
    fig.update_yaxes(title="Wert")
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return lock_axes(fig)


def build_distance_fidelity(latent_pairs, lle_pairs, iso_pairs):
    """Paarabstände der 2-D-Einbettung gegen die Abstände der wahren Faktoren (je auf Mittelwert 1 normiert): auf der Diagonalen ist die Einbettung abstandstreu."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("LLE", "Isomap"))
    for col, (pairs, color) in enumerate(((lle_pairs, ORANGE), (iso_pairs, BLUE)), start=1):
        top = float(max(latent_pairs.max(), pairs.max())) * 1.05
        fig.add_trace(go.Scatter(x=latent_pairs, y=pairs, mode="markers", marker=dict(color=color, size=5, opacity=0.35), hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=GRAY, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=col)
    fig.update_xaxes(title_text="Abstand der wahren Faktoren (normiert)")
    fig.update_yaxes(title_text="Abstand in der Einbettung (normiert)", col=1)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_k_sweep(rows, current_k):
    ks = [r["k"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² der wahren Faktoren", "Abstandstreue und Trustworthiness (LLE)"))
    fig.add_trace(go.Scatter(x=ks, y=[r["r2_lle"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="LLE"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=[r["r2_iso"] for r in rows], mode="lines+markers", line=dict(color=BLUE, width=3, dash="dot"), name="Isomap (Vergleich)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=[r["fid_lle"] for r in rows], mode="lines+markers", line=dict(color=RED, width=3), name="Abstandstreue"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ks, y=[r["trust_lle"] for r in rows], mode="lines+markers", line=dict(color=GREEN, width=3), name="Trustworthiness"), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=current_k, line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.add_vline(x=12.5, line_dash="dash", line_color="rgba(214,138,46,0.6)", annotation_text="k > d = 12", annotation_position="top left", row=1, col=1)
    fig.update_xaxes(title_text="Nachbarn k", type="log", tickvals=[3, 5, 10, 14, 20, 45], ticktext=["3", "5", "10", "14", "20", "45"])
    fig.update_yaxes(range=[0, 1.02])
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_reg_sweep(rows, current_reg):
    rows = [r for r in rows if r["r2_lle"] is not None]
    regs = [r["reg"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² der wahren Faktoren", "Rekonstruktionsfehler der Gewichte"))
    fig.add_trace(go.Scatter(x=regs, y=[r["r2_lle"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="R²"), row=1, col=1)
    fig.add_trace(go.Scatter(x=regs, y=[r["error"] for r in rows], mode="lines+markers", line=dict(color=BLUE, width=3), name="Fehler"), row=1, col=2)
    if current_reg > 0:
        for col in (1, 2):
            fig.add_vline(x=current_reg, line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.update_xaxes(title_text="Regularisierung", type="log")
    fig.update_yaxes(range=[0, 1.02], col=1)
    fig.update_yaxes(type="log", col=2)
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return lock_axes(fig)


def build_out_of_sample(train_embedding, train_color, test_embedding, test_color):
    fig = go.Figure(_scatter(train_embedding, train_color, showscale=True, name="Trainings-Touren", opacity=0.6))
    fig.add_trace(go.Scatter(x=test_embedding[:, 0], y=test_embedding[:, 1], mode="markers", name="neue Touren (über Gewichte eingebettet)", hoverinfo="skip",
                             marker=dict(color=test_color, colorscale="Viridis", cmin=float(train_color.min()), cmax=float(train_color.max()), size=12, symbol="star",
                                         line=dict(width=1.5, color="#14233B"))))
    fig.update_xaxes(title="LLE-Koordinate 1")
    fig.update_yaxes(title="LLE-Koordinate 2")
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_timing(rows):
    ns = np.array([r["n"] for r in rows], dtype=float)
    fig = go.Figure()
    for key, label, color in (("isomap", "Isomap", BLUE), ("lle", "LLE", ORANGE), ("pca", "PCA", GREEN)):
        fig.add_trace(go.Scatter(x=ns, y=[max(r[key], 1e-6) for r in rows], mode="lines+markers", name=label, line=dict(color=color, width=3)))
    fig.update_xaxes(title="Anzahl Touren n", type="log")
    fig.update_yaxes(title="Rechenzeit (s)", type="log")
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)
