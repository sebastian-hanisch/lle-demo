# LLE an Lieferrouten-Kennzahlen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lle-demo.streamlit.app/)**

Drittes Stück der **Dimensionsreduktion-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **LLE (Locally Linear Embedding)** – an einem wachsenden
Beispiel. Vehikel: **dieselben 12 Lieferrouten-Kennzahlen wie in [pca-demo](../pca-demo) und [isomap-demo](../isomap-demo)** (der Generator ist wortgleich kopiert und per Test gegen dessen Ausgabe
eingefroren), erzeugt aus wenigen versteckten Faktoren – dieselbe gekrümmte Fläche, an der PCA scheiterte. Das Isomap-Ergebnis derselben Daten steht als Vergleich daneben.

**Einordnung in die Reihe (die Kanten des Graphen):** LLE ist ein **Kontrast zu Isomap, kein Fix**: beide beheben die Linearitätsschwäche der PCA, aber mit entgegengesetztem Mechanismus –
Isomap **global** (geodätische Abstände über kürzeste Wege), LLE **lokal** (jede Tour wird als Mischung ihrer Nachbarn rekonstruiert, die Einbettung behält die Mischungsverhältnisse).
Beides hat Preis und Nutzen, die die Demo live gegeneinander misst:
```
pca-demo → isomap-demo   (global-geodätisch; Schwäche: Kurzschlüsse, kein Out-of-sample)
pca-demo → lle-demo      (Kontrast zu Isomap, kein Fix: lokal-linear statt global-geodätisch; Schwäche: Regularisierung, Abstände, Rauschen; Stärke: Out-of-sample, Rechenzeit)
pca-demo → t-SNE → UMAP → PaCMAP | Autoencoder   (weitere Äste, noch nicht gebaut)
```

## Was die Demo zeigt

1. **LLE in Aktion** (Schritt-Slider + Abspielen): Nachbarn einer Tour → Rekonstruktionsgewichte (Balken, Rekonstruktion in der 2-D-Ansicht) → Gewichte aller Touren als Graph → Einbettung neben der PCA.
2. **Was LLE gefunden hat – und Isomap auf denselben Daten:** drei Einbettungen nebeneinander, R² der wahren Faktoren, Abstandstreue, Trustworthiness, Rekonstruktionsfehler, Eigenwertspektrum.
3. **📐 Wie stark hängt das Ergebnis von k und der Regularisierung ab?** (live über feste Sweep-Seeds ab 100000, unabhängig vom Demo-Seed): k-Fenster (R², Abstandstreue, Trustworthiness gegen Isomap) und
   Regularisierungs-Sweep, mit Verdict (nicht definiert → Graph zerfällt → Regularisierung zu schwach → k zu klein → Rauschen → kein Vorteil → LLE gewinnt).
4. **📏 Abstände:** Einbettungs-gegen-Faktor-Abstand für LLE und Isomap. **🆕 Out-of-sample:** die letzten 20 % der Touren zurückhalten und über ihre Gewichte einbetten. **⏱️ Rechenzeit** (Knopf, gemessen): LLE gegen Isomap.

Ist die lokale Gram-Matrix singulär und die Regularisierung 0 (k > 12), meldet die Demo **"LLE nicht definiert"**, blendet die LLE-Teile aus und zeigt Isomap und PCA weiter – statt still eine Ersatzlösung zu wählen.

Messwerte (Seed 7, 300 Touren, q = 2, Regularisierung 0.01, k = 14, wenn nicht anders angegeben; die Presets prüfen sie mit Bändern):

| Situation | Messung |
|---|---|
| Gekrümmte Fläche | R² der wahren Faktoren **0.96** (LLE) gegen 0.98 (Isomap) und 0.50 (PCA); Abstandstreue 0.89 gegen 0.96; Trustworthiness 0.99 gegen 1.00 (PCA 0.86) |
| Regularisierung 0.001 statt 0.01 | R² fällt auf **0.66** (Isomap unverändert 0.98) |
| k = 20, Regularisierung 0 | **nicht definiert** (Gram-Matrix Rang 12 < 20); Isomap 0.96 läuft weiter |
| k = 3 | R² **0.44** – unter der PCA (0.50), Abstandstreue 0.36; Isomap mit demselben k: 0.94 |
| Rauschen 0.8 | R² **0.58** gegen 0.84 (Isomap), PCA 0.45 |
| q = 3 Faktoren | dünne Stichprobe, beide fallen ab: LLE 0.56 gegen Isomap 0.47 (PCA 0.19) beim R² – Isomap dafür vorn bei Abstandstreue (0.68 gegen 0.61) und Trustworthiness (0.89 gegen 0.82) |

**k-Fenster** (gekrümmte Daten, Regularisierung 0.01, 3 feste Seeds × 250 Touren): R² der Faktoren 0.31 (k = 3), 0.72–0.75 (k = 4–8, zackig: k = 6 nur 0.56), 0.85 (k = 10),
**0.93–0.95 (k = 12–14)**, danach fallend: 0.85 (k = 20), 0.78 (k = 30), 0.69 (k = 45). Isomap liegt im selben Bereich bei 0.93–0.98 für k = 3…30. Das Fenster ist also **enger und zackiger** als bei Isomap.
Bei Regularisierung 0.001 liegt R² für k = 6…30 durchgehend nur bei 0.66–0.72.

**Regularisierung** (k = 14): R² 0.45 (10⁻⁸), 0.36 (10⁻⁶), 0.44 (10⁻⁴), 0.69 (10⁻³), **0.93 (10⁻²)**, 0.85 (10⁻¹), 0.81 (1). Der Rekonstruktionsfehler der Gewichte steigt dabei monoton von 0.002 auf 400 – **ein kleinerer Rekonstruktionsfehler
heißt nicht, dass die Einbettung besser ist**: bei schwacher Regularisierung passen sich die Gewichte dem Rauschen an.

**Out-of-sample** (10 feste Seeds, Standardeinstellungen, letzte 20 % zurückgehalten): R² der neuen Touren zwischen 0.69 und 0.98, im Median 0.94 – nahe am Trainings-R² desselben Modells, aber mit Ausreißern bei dünn besetzten Stellen.

**Rechenzeit** (lokale Messung, k = 10, ein Lauf je n): n = 100: LLE 4.5 ms, Isomap 2.4 ms (**Isomap ist bei kleinem n schneller** – LLE hat eine Python-Schleife über die Touren); n = 400: 38 ms gegen 132 ms;
n = 1000: **0.28 s gegen 1.7 s** (≈ 6×); PCA bleibt im Bereich 0.1–0.4 ms. Der Vorteil ist ein Faktor, keine andere Größenordnung – beide brauchen eine dichte n×n-Eigenzerlegung.

## Modell und Verfahren

- **Generator** (`lle_scenario.py`): wortgleich aus pca-demo; latente Faktoren, 12 Merkmale in 4 Gruppen, Krümmung `κ·B·h(z)`, Rauschen. LLE arbeitet immer auf z-Werten.
- **LLE** (`lle_algorithm.py`, numpy, ohne sklearn): kNN je Tour (gerichtet) → Rekonstruktionsgewichte `w ∝ G⁻¹𝟙` (Summe 1, lokale Gram-Matrix `G`; Regularisierung `G += reg·tr(G)·I` wie sklearn, bei `reg = 0` und singulärem `G` der
  Fehler `SingularNeighbourhood`) → Eigenzerlegung von `M = (I − W)ᵀ(I − W)`, die Eigenvektoren zu den kleinsten Eigenwerten ohne den konstanten (Eigenwert 0). Out-of-sample: Gewichte einer neuen Tour aus ihren
  k nächsten **Trainings**-Touren, `y = Σ w_j y_j`.
- **Isomap** (`lle_isomap.py`): wortgleich aus isomap-demo kopiert (nur Vergleichspartner; per Test gegen dessen Referenzwerte eingefroren).
- **Auswertung** (`lle_evaluation.py`): R² der wahren Faktoren aus den ersten zwei Koordinaten per quadratischer Regression (eine monotone Umparametrisierung wird nicht bestraft); Abstandstreue = Pearson-Korrelation der
  Paarabstände mit den Faktor-Paarabständen; Trustworthiness (Venna & Kaski, eigene Implementierung); Out-of-sample-Test; k- und Regularisierungs-Sweeps; Zeitmessung.

## Was nicht funktioniert hat / Grenzen

- **Die Standard-Regularisierung von sklearn (0.001) trägt hier nicht:** bei k > d = 12 landet das R² damit bei 0.66–0.72 – deshalb der Default 0.01 und die Regularisierung als eigener Regler.
- **Kein Fenster wie bei Isomap:** LLE ist im k-Fenster zackig (k = 6 schlechter als k = 5); die Demo behauptet deshalb kein "bestes k", sondern zeigt die Kurve.
- **Rauschen und kleines k** brechen LLE stärker als Isomap; bei drei Faktoren sind beide schwach (LLE liegt beim R² knapp vor Isomap, aber hinter ihm bei Abstandstreue und Trustworthiness) – kein Verfahren gewinnt überall.
- **Abstände (Text, gemessen nur an dieser Fläche):** LLE erhält Nachbarschaften, keine Entfernungen; weit entfernte Regionen können beliebig gestaucht oder gedehnt sein.
- **Kein Konvergenzbeweis** gegen die wahre Geometrie im Rauschen.

## Verifikation

- LLE gegen `sklearn.manifold.LocallyLinearEmbedding` (dieselbe Standardisierung, gleiche Regularisierungs-Konvention): Einbettung bis auf Vorzeichen gleich (1e-5), Rekonstruktionsfehler = Summe der Eigenwerte.
- Gewichte summieren sich zu 1 und liegen nur auf den Nachbarn; exakte Rekonstruktion für affin abhängige Nachbarschaften (Handinstanz); `M` positiv semidefinit, konstanter Eigenvektor mit Eigenwert 0; nicht zusammenhängender Graph
  erkannt (mehrere Null-Eigenwerte); `k > d` ohne Regularisierung ⇒ Fehler; Out-of-sample: Trainingspunkte landen bei ihrer eigenen Koordinate.
- Generator bit-identisch zu pca-demo und Isomap-Kopie bit-identisch zu isomap-demo (eingefrorene Referenzwerte, Geodäten gegen `scipy`-Dijkstra); Trustworthiness gegen `sklearn.manifold.trustworthiness` (1e-9).
- Sweeps über feste Seeds deterministisch; Verdict-Codes; alle 6 Presets in kalibrierten Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt, Randgrößen, Schritt-Zustand, "nicht definiert"-Zustand),
  Achsensperre aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 k-/Regularisierungs-Sweep, Abstände, Out-of-sample, Rechenzeit, Mathe |
| `lle_algorithm.py` | LLE von Grund auf (Nachbarn, Gewichte, Einbettung, Out-of-sample) |
| `lle_isomap.py` | Isomap-Kern (wortgleich aus isomap-demo, Vergleichspartner) |
| `lle_scenario.py`, `lle_constants.py` | Lieferrouten-Generator (wortgleich aus pca-demo), Konstanten, Presets |
| `lle_evaluation.py` | Kennzahlen, Sweeps, Verdict, Zeitmessung |
| `lle_presets.py`, `lle_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | sklearn-/scipy-Kreuzvergleiche, Generator-Referenz, Auswertung, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Dimensionsreduktion: von PCA bis Autoencoder](https://sebastianhanisch.net/konzepte-dimensionsreduktion.html).
