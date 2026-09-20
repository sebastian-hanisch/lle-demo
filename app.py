"""LLE an Lieferrouten-Kennzahlen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - LLE (Locally Linear
Embedding) - und lässt stattdessen das Beispiel wachsen. Drittes Stück der Dimensionsreduktion-Linie der "Konzepte"-Reihe: LLE ist ein KONTRAST zu
Isomap (kein Fix): beide beheben die Linearitätsschwäche der PCA, Isomap global über geodätische Abstände, LLE lokal über Rekonstruktionsgewichte.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import lle_constants as C
from lle_evaluation import analyse, k_sweep, make_dataset, reg_sweep, timing_sweep, verdict
from lle_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from lle_visualization import (
    build_distance_fidelity,
    build_embedding,
    build_k_sweep,
    build_out_of_sample,
    build_reg_sweep,
    build_spectrum,
    build_timing,
    build_view,
    build_weights_bar,
)

st.set_page_config(page_title="LLE – Sebastian Hanisch", layout="wide")

STEP_LABELS = {
    1: "1 · Nachbarn einer Tour",
    2: "2 · Rekonstruktionsgewichte",
    3: "3 · Gewichte aller Touren",
    4: "4 · Einbettung (Eigenvektoren)",
}


def _reg_label(value):
    return "0 (keine)" if value == 0 else f"{value:g}"


@st.cache_data(show_spinner=False)
def _dataset(n_tours, q, curvature, noise, seed):
    return make_dataset(n_tours, q, curvature, noise, seed)


@st.cache_data(show_spinner=False)
def _analysis(n_tours, q, curvature, noise, seed, k, reg):
    return analyse(make_dataset(n_tours, q, curvature, noise, seed), k, reg)


@st.cache_data(show_spinner=False)
def _k_sweep(q, curvature, noise, reg):
    return k_sweep(q, curvature, noise, reg)


@st.cache_data(show_spinner=False)
def _reg_sweep(q, curvature, noise, k):
    return reg_sweep(q, curvature, noise, k)


st.title("🧵 LLE an Lieferrouten-Kennzahlen")
st.markdown(
    """
Dieselben **12 Kennzahlen je Lieferroute** wie in der PCA- und der Isomap-Demo - erzeugt aus wenigen versteckten Faktoren, aber mit **gekrümmter** Struktur, an der
die PCA scheiterte. **LLE** (Locally Linear Embedding) löst dasselbe Problem wie Isomap - aber mit dem entgegengesetzten Ansatz: statt globaler Wege-Abstände
schaut es nur **lokal**. Jede Tour wird als **gewichtete Mischung ihrer *k* nächsten Nachbarn** rekonstruiert; die Einbettung behält genau diese Mischungsverhältnisse bei.
Was dabei gewonnen und was verloren geht - Regularisierung, Abstandstreue, neue Touren, Rechenzeit - zeigt die Demo direkt gegen Isomap. Wie das Verfahren funktioniert,
erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - drittes Stück der "
    "Dimensionsreduktion-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel: LLE ist ein **Kontrast zu Isomap** (kein Fix): "
    "lokal-linear statt global-geodätisch, mit eigenen Schwächen und einer eigenen Stärke."
)

with st.expander("So funktioniert LLE", expanded=True):
    st.markdown(
        """
LLE (Roweis & Saul, 2000) besteht aus drei Schritten:

1. **Nachbarn**: jede Tour bekommt ihre *k* nächsten Nachbarn (im Merkmalsraum, in z-Werten).
2. **Rekonstruktionsgewichte**: jede Tour wird als **Mischung ihrer Nachbarn** geschrieben (die Gewichte summieren sich zu 1). Die Gewichte beschreiben die **lokale Geometrie**
   der Fläche um die Tour - unabhängig davon, wo im Raum sie liegt. Mit mehr Nachbarn als Merkmalen (*k* > 12) gibt es unendlich viele Lösungen: dann braucht LLE eine
   **Regularisierung**, sonst ist es nicht definiert.
3. **Einbettung**: gesucht sind 2-D-Koordinaten, in denen jede Tour **dieselbe Mischung** ihrer Nachbarn ist wie im Merkmalsraum. Das ist ein Eigenwertproblem - die Koordinaten sind die
   Eigenvektoren zu den kleinsten Eigenwerten (den konstanten ausgenommen).

Gegenüber Isomap gilt: LLE braucht **keine kürzesten Wege** (deutlich schneller) und kann **neue Touren** über ihre Gewichte einbetten - erhält aber **keine Abstände**, nur
Nachbarschaften, und reagiert empfindlich auf Regularisierung, ein zu kleines *k* und Rauschen. Die Regler links zeigen jede dieser Eigenschaften live.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider("Anzahl Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=50)
    q = st.slider(
        "Wahre Anzahl versteckter Faktoren (q)", *bounds("q_slider"), key="q_slider",
        help="So viele echte Einflussgrößen erzeugen die 12 Kennzahlen. Mit mehr Faktoren wird die Fläche höherdimensional - bei gleich vielen Touren wird die Stichprobe dünner.",
    )
    curvature = st.slider(
        "Krümmung", *bounds("curvature_slider"), key="curvature_slider", step=0.05,
        help="0 = die Kennzahlen hängen linear von den Faktoren ab (dann hat LLE keinen Vorteil vor der PCA). Größer = die Touren liegen auf einer zunehmend gebogenen Fläche.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen je Kennzahl. Rauschen macht lokale Nachbarschaften unzuverlässig - LLE, das nur lokal schaut, leidet besonders.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Lokale Nachbarschaften**")
    k = st.slider(
        "Nachbarn k", *bounds("k_slider"), key="k_slider",
        help="Jede Tour wird aus ihren k nächsten Nachbarn rekonstruiert. Zu klein: die Nachbarschaft spannt die Fläche nicht auf. Über 12 (Zahl der Merkmale): die Gewichte sind nur mit Regularisierung eindeutig.",
    )
    reg = st.select_slider(
        "Regularisierung", options=C.REG_CHOICES, key="reg_select", format_func=_reg_label,
        help="Zuschlag auf die Diagonale der lokalen Gram-Matrix (relativ zu ihrer Spur). Bei k > 12 nötig; 0 = keine Regularisierung - dann ist LLE für k > 12 nicht definiert.",
    )

    st.button("🎲 Neue Touren generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren.")

sync_query_params(n_tours, q, curvature, noise, k, reg, seed)

params = (int(n_tours), int(q), float(curvature), float(noise), int(seed))
with st.spinner("Berechne Nachbarn, Rekonstruktionsgewichte und Einbettungen..."):
    dataset = _dataset(*params)
    analysis = _analysis(*params, int(k), float(reg))
model = analysis.lle
z_color = dataset.z[:, 0]
iso_idx = analysis.iso_indices
data_key = params + (int(k), float(reg))
level, code, vd = verdict(analysis, dataset, int(k), float(reg))

if model is None:
    st.error(
        f"⛔ **LLE ist mit diesen Einstellungen nicht definiert:** {vd['message']}. Bei k = {vd['k']} Nachbarn und nur {vd['d']} Merkmalen hat jede Tour unendlich viele "
        "Rekonstruktionsgewichte - die Demo wählt keine stille Ersatzlösung. Regularisierung erhöhen (Regler links) oder k auf höchstens 12 senken. Isomap und PCA laufen unten weiter."
    )

# --- LLE in Aktion -------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 LLE in Aktion")
if model is None:
    st.info("Die Schritte brauchen definierte Rekonstruktionsgewichte - siehe Fehlermeldung oben.")
else:
    st.caption(
        "Die 2-D-Ansicht zeigt die Touren in den ersten beiden Hauptkomponenten (Farbe = versteckter Faktor 1) - nur als Zeichenfläche; LLE selbst rechnet in allen 12 Dimensionen. "
        "Wo die Fläche gebogen ist, liegen in dieser Ansicht Teile übereinander."
    )
    if "lle_step" not in st.session_state or st.session_state.get("lle_step_owner") != data_key:
        st.session_state["lle_step"] = 1
        st.session_state["lle_step_owner"] = data_key
    step_col, play_col = st.columns([5, 1])
    with step_col:
        step = st.select_slider("Schritt", options=list(STEP_LABELS), key="lle_step", format_func=lambda s: STEP_LABELS[s])
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")

    view = analysis.pca_2d
    centre = view.mean(0)
    focus = int(np.argmin(((view - centre) ** 2).sum(1)))
    nbrs = model.neighbours[focus]
    w_focus = model.W[focus, nbrs]
    reconstruction = w_focus @ view[nbrs]
    edge_step = max(1, len(view) // 100)
    edges = [(i, int(j)) for i in range(0, len(view), edge_step) for j in model.neighbours[i]]

    view_slot = st.empty()

    def _render(current_step):
        if current_step == 1:
            view_slot.plotly_chart(build_view(view, z_color, focus=focus, neighbours=nbrs), width="stretch", key=f"lle_view_{current_step}")
        elif current_step == 2:
            with view_slot.container():
                c1, c2 = st.columns([3, 2])
                c1.plotly_chart(build_view(view, z_color, focus=focus, neighbours=nbrs, weights=w_focus, reconstruction=reconstruction), width="stretch", key=f"lle_view_{current_step}")
                c2.markdown("**Gewichte der Nachbarn**")
                c2.plotly_chart(build_weights_bar(w_focus), width="stretch", key="lle_weights_bar")
        elif current_step == 3:
            view_slot.plotly_chart(build_view(view, z_color, edges=edges), width="stretch", key=f"lle_view_{current_step}")
        else:
            with view_slot.container():
                c1, c2 = st.columns(2)
                c1.markdown("**LLE: Einbettung aus den Rekonstruktionsgewichten**")
                c1.plotly_chart(build_embedding(analysis.lle_2d, z_color, "LLE-Koordinate 1", "LLE-Koordinate 2"), width="stretch", key="lle_embed_step")
                c2.markdown("**Zum Vergleich: PCA**")
                c2.plotly_chart(build_embedding(analysis.pca_2d, z_color, "PC1", "PC2"), width="stretch", key="pca_embed_step")

    if auto_play:
        for s in STEP_LABELS:
            _render(s)
            time.sleep(1.0)
        step = 4
    else:
        _render(step)

    if step == 1:
        st.caption(f"Die gewählte Tour (Stern, nahe der Mitte) und ihre **k = {model.k}** nächsten Nachbarn (orange) im Merkmalsraum. Nur diese Nachbarn gehen in ihre Rekonstruktion ein.")
    elif step == 2:
        st.caption(
            f"Die Tour wird als Mischung ihrer {model.k} Nachbarn geschrieben (Summe der Gewichte = 1; {int((w_focus < 0).sum())} davon negativ - die Tour darf außerhalb der Nachbarn liegen). "
            f"Die grüne Raute ist die Mischung in dieser Ansicht; ihr Abstand zum Stern ist der Rekonstruktionsfehler dieser Tour: {model.errors[focus]:.4f} (z-Einheiten², alle 12 Dimensionen)."
        )
    elif step == 3:
        st.caption(
            f"Dasselbe für jede Tour ({len(edges) // model.k} der {model.n} sind als Kanten von der Tour zu ihren Nachbarn gezeichnet). Zusammen ergeben die Gewichte eine Matrix W; "
            f"der gesamte Rekonstruktionsfehler ist {model.reconstruction_error:.3f}. Ist der Graph nicht zusammenhängend, zerfällt die Einbettung in unabhängige Teile."
        )
    else:
        st.caption("Farbe = versteckter Faktor 1. Verläuft sie in der LLE-Einbettung glatt und ohne Überlappung, hat LLE die Fläche entrollt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was LLE gefunden hat - und Isomap auf denselben Daten")
if model is not None and not model.connected:
    st.warning(
        f"⚠️ Der Nachbarschaftsgraph ist mit k = {model.k} **nicht zusammenhängend** ({int(model.components.max()) + 1} Teile): für jede Komponente gibt es einen eigenen Null-Eigenwert - "
        "die Koordinaten sind mehrdeutig und die Teile liegen unverbunden nebeneinander."
    )
if len(iso_idx) < dataset.n:
    st.warning(f"⚠️ Der Isomap-Graph ist mit k = {int(k)} nicht zusammenhängend: nur {len(iso_idx)} von {dataset.n} Touren sind in der Isomap-Einbettung enthalten (siehe Isomap-Demo).")

m1, m2, m3, m4 = st.columns(4)
if model is not None:
    m1.metric("R² der wahren Faktoren", f"{analysis.r2['lle']:.2f}", delta=f"{analysis.r2['lle'] - analysis.r2['isomap']:+.2f} ggü. Isomap", delta_color="normal",
              help="Wie gut lassen sich die versteckten Faktoren aus den ersten zwei Koordinaten zurückgewinnen (quadratische Regression). Isomap mit derselben Messung im Delta.")
    m2.metric("Abstandstreue", f"{analysis.fidelity['lle']:.2f}", delta=f"{analysis.fidelity['lle'] - analysis.fidelity['isomap']:+.2f} ggü. Isomap", delta_color="normal",
              help="Korrelation der Paarabstände in der Einbettung mit den Paarabständen der wahren Faktoren. LLE erhält Nachbarschaften, keine Abstände.")
    m3.metric("Trustworthiness", f"{analysis.trust['lle']:.2f}", delta=f"{analysis.trust['lle'] - analysis.trust['isomap']:+.2f} ggü. Isomap", delta_color="normal",
              help=f"Nachbarschaft erhalten: Anteil der Nachbarn in der 2-D-Einbettung, die auch im Originalraum Nachbarn sind (k = {C.TRUST_NEIGHBORS}); 1 = perfekt.")
    m4.metric("Rekonstruktionsfehler", f"{model.reconstruction_error:.3f}", help="Summe über alle Touren: wie gut die Nachbarn-Mischung die Tour im Merkmalsraum trifft (kleiner = besser).")
else:
    m1.metric("R² der wahren Faktoren (Isomap)", f"{analysis.r2['isomap']:.2f}")
    m2.metric("Abstandstreue (Isomap)", f"{analysis.fidelity['isomap']:.2f}")
    m3.metric("Trustworthiness (Isomap)", f"{analysis.trust['isomap']:.2f}")
    m4.metric("LLE", "nicht definiert")

e1, e2, e3 = st.columns(3)
with e1:
    st.markdown("**LLE**")
    if model is not None:
        st.plotly_chart(build_embedding(analysis.lle_2d, z_color, "LLE-Koordinate 1", "LLE-Koordinate 2"), width="stretch", key="lle_embedding")
    else:
        st.info("nicht definiert")
with e2:
    st.markdown("**Isomap (Vergleich)**")
    st.plotly_chart(build_embedding(analysis.iso_2d, z_color[iso_idx], "Isomap-Koordinate 1", "Isomap-Koordinate 2"), width="stretch", key="iso_embedding")
with e3:
    st.markdown("**PCA (Vergleich)**")
    st.plotly_chart(build_embedding(analysis.pca_2d, z_color, "PC1", "PC2"), width="stretch", key="pca_embedding")

table = {"Verfahren": ["LLE", "Isomap", "PCA"]}
for label, source in (("R² der Faktoren", analysis.r2), ("Abstandstreue", analysis.fidelity), ("Trustworthiness", analysis.trust)):
    table[label] = ["–" if source["lle"] is None else f"{source['lle']:.2f}", f"{source['isomap']:.2f}", f"{source['pca']:.2f}"]
st.table(table)

if model is not None:
    st.markdown("**Eigenwertspektrum von M = (I − W)ᵀ(I − W)**")
    st.plotly_chart(build_spectrum(model.eigenvalues, 2), width="stretch", key="lle_spectrum")
    st.caption(
        "Der erste Eigenwert (grau, ≈ 0) gehört zum konstanten Vektor und wird übersprungen; die nächsten beiden (orange) liefern die Koordinaten. Je kleiner sie sind, desto besser lassen sich "
        "die Rekonstruktionsgewichte in 2 Dimensionen halten. Mehrere Nullen ganz links heißen: der Graph zerfällt."
    )

st.markdown("---")

# --- k-Fenster und Regularisierung -----------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von k und der Regularisierung ab?")
st.markdown(
    """
*k* und die **Regularisierung** sind LLEs wichtigste Regler - und beide haben ein **Fenster**: bei zu kleinem *k* spannen die Nachbarn die Fläche nicht auf, bei *k* > 12 hängt das
Ergebnis von der Regularisierung ab, und Rauschen macht lokale Nachbarschaften unzuverlässig. Live für Ihr aktuelles Szenario über **feste Sweep-Seeds** (unabhängig vom Demo-Seed) geprüft, nicht behauptet:
"""
)
if code == "undefined":
    st.info("Für die aktuelle Kombination ist LLE nicht definiert (siehe oben); die Kurven zeigen LLE deshalb nur, wo es definiert ist.")
elif code == "disconnected":
    st.warning(f"⚠️ **Der Graph zerfällt** (k = {vd['k']}): {vd['components']} Komponenten - ein größeres k verbindet die Teile.")
elif code == "reg_weak":
    st.warning(
        f"⚠️ **Regularisierung zu schwach**: bei k = {vd['k']} > {vd['d']} Merkmalen sind {vd['singular'] * 100:.0f} % der lokalen Gram-Matrizen singulär, Regularisierung {vd['reg']:g} reicht nicht - "
        f"R² der Faktoren nur {vd['r2_lle']:.2f} (Isomap {vd['r2_iso']:.2f}, PCA {vd['r2_pca']:.2f}). Mit 0.01 wird es deutlich besser (Kurve unten)."
    )
elif code == "k_small":
    st.warning(
        f"⚠️ **Zu wenige Nachbarn**: bei k = {vd['k']} spannen die Nachbarschaften die Fläche nicht auf - R² der Faktoren nur {vd['r2_lle']:.2f} (PCA {vd['r2_pca']:.2f}), "
        f"Isomap mit demselben k kommt auf {vd['r2_iso']:.2f}."
    )
elif code == "noise":
    st.warning(
        f"⚠️ **Rauschen**: bei diesem Rauschen sind lokale Nachbarschaften unzuverlässig - R² der Faktoren nur {vd['r2_lle']:.2f} (Isomap {vd['r2_iso']:.2f}, PCA {vd['r2_pca']:.2f}). "
        "Isomap mittelt über ganze Wege und ist hier robuster."
    )
elif code == "no_advantage":
    st.info(f"ℹ️ **Kein Vorteil vor der PCA**: die Daten sind gerade (Krümmung 0) - R² der Faktoren {vd['r2_lle']:.2f} (LLE) gegen {vd['r2_pca']:.2f} (PCA).")
elif code == "lle_wins":
    st.success(
        f"✅ **LLE entrollt die Fläche**: R² der Faktoren {vd['r2_lle']:.2f} gegen {vd['r2_pca']:.2f} bei der PCA (Isomap: {vd['r2_iso']:.2f}); Abstandstreue {vd['fid_lle']:.2f} gegen {vd['fid_iso']:.2f} bei Isomap - "
        "Nachbarschaften stimmen, Abstände weniger."
    )
else:
    st.info(f"LLE erreicht R² {vd['r2_lle']:.2f} (Isomap {vd['r2_iso']:.2f}, PCA {vd['r2_pca']:.2f}) - kein klarer Gewinn und kein klarer Bruch.")

st.markdown("**Nachbarn k** (Regularisierung wie links eingestellt)")
k_rows = _k_sweep(int(q), float(curvature), float(noise), float(reg))
if all(r["r2_lle"] is None for r in k_rows):
    st.info("Mit dieser Regularisierung ist LLE für keines der getesteten k definiert.")
else:
    st.plotly_chart(build_k_sweep([r for r in k_rows if r["r2_lle"] is not None], int(k)), width="stretch", key="k_sweep")
    ok = [r["k"] for r in k_rows if r["r2_lle"] is not None and r["r2_lle"] >= 0.9]
    st.caption(
        f"Gleiche Einstellungen (q = {dataset.q}, Krümmung {curvature:.2f}, Rauschen {noise:.2f}, Regularisierung {_reg_label(float(reg))}), nur k wächst; Mittel über {len(C.SWEEP_SEEDS)} feste Seeds mit je {C.SWEEP_N_TOURS} Touren. "
        + (f"R² der Faktoren ≥ 0.9: k = {min(ok)} … {max(ok)} (getestet: {', '.join(str(r['k']) for r in k_rows)})." if ok else "Für diese Einstellungen erreicht kein getestetes k ein R² von 0.9.")
    )

st.markdown(f"**Regularisierung** (bei k = {int(k)})")
reg_rows = _reg_sweep(int(q), float(curvature), float(noise), int(k))
if int(k) <= 12:
    st.caption(f"Bei k = {int(k)} ≤ 12 sind die lokalen Gram-Matrizen regulär - die Regularisierung ändert wenig. Interessant wird sie ab k = 13; die Kurve zeigt trotzdem den Verlauf.")
if all(r["r2_lle"] is None for r in reg_rows):
    st.info("Für dieses k ist LLE bei keiner getesteten Regularisierung definiert.")
else:
    st.plotly_chart(build_reg_sweep(reg_rows, float(reg)), width="stretch", key="reg_sweep")
    best = max((r for r in reg_rows if r["r2_lle"] is not None), key=lambda r: r["r2_lle"])
    st.caption(
        f"Mittel über {len(C.SWEEP_SEEDS)} feste Seeds mit je {C.SWEEP_N_TOURS} Touren. Beste getestete Regularisierung: {best['reg']:g} (R² {best['r2_lle']:.2f}). "
        "Der Rekonstruktionsfehler (rechts) sinkt mit kleinerer Regularisierung immer weiter - die Einbettung wird dabei aber schlechter: die Gewichte passen sich dem Rauschen an. Zu große Regularisierung verwischt dagegen die lokale Geometrie."
    )

st.markdown("---")

# --- Verzerrung der Abstände -----------------------------------------------------------------------------------------------

st.markdown("## 📏 LLE erhält Nachbarschaften, keine Abstände")
if model is None:
    st.info("Nicht verfügbar, solange LLE nicht definiert ist.")
else:
    rng = np.random.default_rng(0)
    m_iso = len(iso_idx)
    pa = rng.integers(0, m_iso, size=min(1500, m_iso * (m_iso - 1) // 2))
    pb = rng.integers(0, m_iso, size=len(pa))
    keep = pa != pb
    pa, pb = pa[keep], pb[keep]
    ga, gb = iso_idx[pa], iso_idx[pb]

    def _norm(d):
        return d / d.mean()

    latent = _norm(np.linalg.norm(dataset.z[ga] - dataset.z[gb], axis=1))
    lle_d = _norm(np.linalg.norm(analysis.lle_2d[ga] - analysis.lle_2d[gb], axis=1))
    iso_d = _norm(np.linalg.norm(analysis.iso_2d[pa] - analysis.iso_2d[pb], axis=1))
    st.plotly_chart(build_distance_fidelity(latent, lle_d, iso_d), width="stretch", key="distance_fidelity")
    st.caption(
        f"Jeder Punkt ein Tourenpaar: Abstand in der 2-D-Einbettung gegen den Abstand der wahren Faktoren. Auf der gestrichelten Diagonale wäre die Einbettung abstandstreu. "
        f"Korrelation: LLE {analysis.fidelity['lle']:.2f}, Isomap {analysis.fidelity['isomap']:.2f}. LLE stellt nur sicher, dass Nachbarn Nachbarn bleiben (Trustworthiness "
        f"{analysis.trust['lle']:.2f}); wie weit entfernte Regionen auseinander liegen, ist ihm freigestellt."
    )

st.markdown("---")

# --- Out-of-sample ---------------------------------------------------------------------------------------------------------

st.markdown("## 🆕 Neue Touren einbetten (Out-of-sample)")
st.caption(
    "Isomap kennt keine Abbildung für neue Touren. LLE schon: eine neue Tour bekommt ihre Rekonstruktionsgewichte aus den **k nächsten Trainings-Touren** und wird als dieselbe Mischung der "
    f"Trainings-Koordinaten eingebettet - ohne Neuberechnung. Test: die letzten {C.HOLDOUT_FRACTION * 100:.0f} % der Touren zurückhalten, LLE auf den übrigen trainieren."
)
oos = analysis.oos
if oos is None:
    st.info("Nicht verfügbar, solange LLE nicht definiert ist.")
else:
    st.plotly_chart(
        build_out_of_sample(oos["model"].embedding[:, :2], dataset.z[oos["train"], 0], oos["y_test"][:, :2], dataset.z[oos["test"], 0]), width="stretch", key="oos_plot",
    )
    st.caption(
        f"Sterne = zurückgehaltene Touren, eingebettet über ihre Gewichte. R² der wahren Faktoren (über eine auf den Trainings-Touren angepasste Abbildung Einbettung → Faktoren): "
        f"**{oos['r2_test']:.2f}** für die neuen, {oos['r2_train']:.2f} für die Trainings-Touren. Über 10 feste Seeds (Standardeinstellungen) lag das R² der neuen Touren zwischen 0.69 und 0.98, im Median bei 0.94 - "
        "fällt eine neue Tour in eine dünn besetzte Stelle, ist ihre Einbettung ungenauer (🎲 würfelt einen anderen Datensatz)."
    )

st.markdown("---")

# --- Rechenzeit ------------------------------------------------------------------------------------------------------------

st.markdown("## ⏱️ Rechenzeit: LLE gegen Isomap")
st.caption(
    "Isomap braucht kürzeste Wege zwischen allen Paaren (kubisch) und eine dichte n×n-Eigenzerlegung. LLE löst nur je Tour ein kleines k×k-System, braucht aber ebenfalls eine n×n-Eigenzerlegung - "
    "der Vorteil ist ein Faktor, keine andere Größenordnung - und erst bei größerem n sichtbar: bei wenigen Touren ist Isomap noch schneller."
)
if "timing_rows" not in st.session_state:
    if st.button("⏱️ Rechenzeit messen (ca. 10 s)", key="timing_start", help=f"Misst LLE, Isomap (jeweils k = {C.TIMING_K}) und PCA für n = {', '.join(str(n) for n in C.TIMING_NS)} auf diesem Rechner."):
        with st.spinner("Messe..."):
            st.session_state["timing_rows"] = timing_sweep()
        st.rerun()
else:
    rows = st.session_state["timing_rows"]
    st.plotly_chart(build_timing(rows), width="stretch", key="timing_chart")
    st.table({
        "Touren n": [r["n"] for r in rows],
        "LLE": [f"{r['lle']:.3f} s" for r in rows],
        "Isomap": [f"{r['isomap']:.3f} s" for r in rows],
        "PCA": [f"{r['pca'] * 1000:.2f} ms" for r in rows],
        "Isomap / LLE": [f"{r['isomap'] / max(r['lle'], 1e-9):.1f}×" for r in rows],
    })
    st.caption(f"Gemessen auf diesem Rechner (Wandzeit, k = {C.TIMING_K}, ein Lauf je n). Die genauen Faktoren hängen von Rechner und Zwischenspeichern ab.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Rekonstruktionsgewichte.** Gegeben $n$ Punkte $x_i \in \mathbb{R}^d$ (hier: z-Werte der 12 Kennzahlen) und die $k$ nächsten Nachbarn $\mathcal{N}(i)$ jeder Tour. LLE sucht Gewichte $w_{ij}$ mit

$$
\min_{W} \ \varepsilon(W) = \sum_i \Big\lVert x_i - \sum_{j \in \mathcal{N}(i)} w_{ij}\, x_j \Big\rVert^2 \quad \text{unter} \quad \sum_j w_{ij} = 1 .
$$

Mit der lokalen Gram-Matrix $G^{(i)}_{jl} = (x_j - x_i)^\top (x_l - x_i)$ und einem Lagrange-Multiplikator für die Nebenbedingung folgt $w^{(i)} \propto \big(G^{(i)}\big)^{-1}\mathbf{1}$, normiert auf Summe 1.
Wegen $\sum_j w_{ij} = 1$ sind die Gewichte invariant gegen Verschiebung, Drehung und Skalierung der Nachbarschaft - sie beschreiben nur ihre lokale Geometrie.

**Regularisierung.** Bei $k > d$ hat $G^{(i)}$ höchstens Rang $d$ und ist singulär: die Lösung ist nicht eindeutig. Die Demo ersetzt $G^{(i)}$ durch $G^{(i)} + r \cdot \operatorname{tr}(G^{(i)})\, I$ (Konvention wie in
scikit-learn); für $r = 0$ und $k > d$ meldet sie den Fehler, statt still zu reparieren. $r$ verschiebt die Gewichte in Richtung gleicher Gewichte $1/k$ - zu große $r$ verwischt die lokale Geometrie.

**Einbettung.** Bei festem $W$ minimieren die Koordinaten $y_i \in \mathbb{R}^m$ den Fehler $\Phi(Y) = \sum_i \lVert y_i - \sum_j w_{ij} y_j \rVert^2 = \operatorname{tr}\big(Y^\top M Y\big)$ mit $M = (I - W)^\top (I - W)$,
unter $\tfrac1n Y^\top Y = I$ und $\sum_i y_i = 0$. Lösung: die Eigenvektoren von $M$ zu den $m$ kleinsten Eigenwerten **ohne** den konstanten Eigenvektor (Eigenwert 0, da $W\mathbf{1} = \mathbf{1}$).

**Out-of-sample.** Eine neue Tour $x$ bekommt aus ihren $k$ nächsten Trainings-Touren Gewichte $w_j$ wie oben und die Koordinate $y = \sum_j w_j\, y_j$.

**Grenzen.** (1) *Regularisierung*: bei $k > d$ entscheidet sie über das Ergebnis (Demo: R² 0.66 bei $r = 10^{-3}$ gegen 0.96 bei $10^{-2}$). (2) *Keine Abstände*: LLE erhält Nachbarschaften, nicht Entfernungen -
weit entfernte Regionen können beliebig gestaucht oder gedehnt werden. (3) *Zusammenhang*: ein nicht zusammenhängender kNN-Graph erzeugt mehrere Eigenwerte 0, die Einbettung ist dann mehrdeutig.
(4) *Rauschen und kleines $k$*: lokale Nachbarschaften werden unzuverlässig bzw. spannen die Fläche nicht auf. (5) Es gibt keinen einfachen Konvergenzbeweis gegen die wahre Geometrie im Rauschen.

**Trustworthiness** (Venna & Kaski, 2001): $T = 1 - \frac{2}{nk(2n - 3k - 1)} \sum_i \sum_{j \in U_i} (r(i,j) - k)$ mit $U_i$ = Nachbarn in der Einbettung, die im Originalraum keine sind, und $r(i,j)$ ihrem Originalrang.
**Abstandstreue** = Pearson-Korrelation der Paarabstände der 2-D-Einbettung mit den Paarabständen der wahren Faktoren.

Implementiert in `lle_algorithm.py` (Nachbarn, Gewichte, Einbettung, Out-of-sample), `lle_isomap.py` (Isomap-Kern, wortgleich aus isomap-demo, nur als Vergleich), `lle_scenario.py` (Lieferrouten-Generator,
wortgleich aus pca-demo) und `lle_evaluation.py` (Kennzahlen, Sweeps, Verdict, Zeitmessung).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
