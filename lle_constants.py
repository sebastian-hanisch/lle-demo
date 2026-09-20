"""Defaults, Slider-Grenzen und Presets für die LLE-Demo. Merkmale und Erzeugungs-Konstanten sind wortgleich aus pca-demo übernommen
(dieselben Lieferrouten - dieselbe gekrümmte Fläche, an der PCA scheiterte); alles Übrige ist neu."""

# --- Merkmale: 12 Kennzahlen je Tour in 4 Gruppen zu je 3 (Name, Einheit, Mittelwert, typische Streuung in Einheiten) ------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_FEATURES = len(FEATURES)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
FEATURE_LABELS = tuple(f"{f[0]} [{f[1]}]" for f in FEATURES)
GROUPS = ("Größe", "Zeitdruck", "Verkehr", "Sonderfälle")     # je 3 aufeinanderfolgende Merkmale
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_FEATURES))

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 100, 600
DEFAULT_Q = 2
Q_MIN, Q_MAX = 1, 4
DEFAULT_CURVATURE = 1.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_K = 14
K_MIN, K_MAX = 2, 60
REG_CHOICES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
DEFAULT_REG = 1e-2
DEFAULT_SEED = 7

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
OUTLIER_SCALE = 10.0                   # Sonderfahrten: latenter Faktor um diesen Faktor vergrößert
CROSS_LOADING = 0.15                   # kleine Querladungen zwischen Merkmalsgruppen
WITHIN_LOADINGS = (0.95, 0.9, 0.85)    # Ladung der drei Merkmale einer Gruppe auf ihren Faktor
CURVATURE_FREQUENCY = 1.6              # Frequenz der sin/cos-Terme der Krümmung
CURVATURE_AMPLITUDE = 2.0              # Länge jeder Spalte der Krümmungsmatrix (in z-Einheiten bei Krümmung 1)
LAYOUT_SEED = 20240915                 # feste Ladungs- und Krümmungsmatrizen (unabhängig vom Seed der Touren)


# --- Auswertung --------------------------------------------------------------------------------------------------------
N_COMPONENTS_MAX = 10
TRUST_NEIGHBORS = 10
SWEEP_SEEDS = tuple(100_000 + i for i in range(3))                 # feste Sweep-Seeds, unabhängig vom Demo-Seed
SWEEP_KS = (3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 30, 45)
SWEEP_REGS = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SWEEP_N_TOURS = 250
TIMING_NS = (100, 200, 400, 600, 800, 1000)
TIMING_K = 10
HOLDOUT_FRACTION = 0.2

_BASE = {"n_tours": DEFAULT_N_TOURS, "q": 2, "curvature": 1.0, "noise": DEFAULT_NOISE, "k": DEFAULT_K, "reg": DEFAULT_REG, "seed": DEFAULT_SEED}
PRESETS = {
    "Gekrümmte Fläche: LLE entrollt": {**_BASE},
    "Regularisierung zu schwach": {**_BASE, "reg": 1e-3},
    "Ohne Regularisierung: nicht definiert": {**_BASE, "k": 20, "reg": 0.0},
    "Zu kleines k": {**_BASE, "k": 3},
    "Rauschen: Isomap ist robuster": {**_BASE, "noise": 0.8},
    "Drei Faktoren: dünne Stichprobe": {**_BASE, "q": 3},
}
PRESET_HELP = {
    "Gekrümmte Fläche: LLE entrollt": "Dieselbe gebogene Fläche wie in der PCA- und der Isomap-Demo: LLE gewinnt die versteckten Faktoren fast so gut zurück wie Isomap (R² ≈ 0.96 gegen 0.98), die PCA nur zur Hälfte (≈ 0.50) - aber die Abstände bleiben ungenauer (Abstandstreue 0.89 gegen 0.96).",
    "Regularisierung zu schwach": "Bei 14 Nachbarn und nur 12 Merkmalen ist die lokale Gram-Matrix singulär - die Regularisierung entscheidet über das Ergebnis: mit 0.001 fällt das R² auf etwa 0.66 (mit 0.01 sind es 0.96), Isomap bleibt bei 0.98.",
    "Ohne Regularisierung: nicht definiert": "Mit 20 Nachbarn und Regularisierung 0 gibt es für jede Tour unendlich viele Rekonstruktionsgewichte: LLE ist nicht definiert. Die Demo meldet es, statt still eine Ersatzlösung zu wählen - Isomap (0.96) und PCA laufen weiter.",
    "Zu kleines k": "Mit nur drei Nachbarn je Tour spannen die Nachbarschaften die Fläche nicht auf: das R² der Faktoren fällt auf etwa 0.44 - unter das der PCA (0.50), während Isomap mit demselben k noch 0.94 erreicht.",
    "Rauschen: Isomap ist robuster": "Mit viel Rauschen sind lokale Nachbarschaften unzuverlässig: LLE erreicht nur noch etwa 0.58, Isomap 0.84 (PCA 0.45).",
    "Drei Faktoren: dünne Stichprobe": "Bei drei versteckten Faktoren ist die Fläche höherdimensional, 300 Touren tasten sie dünn ab: beide Verfahren fallen ab (LLE R² ≈ 0.56, Isomap ≈ 0.47, PCA ≈ 0.19). Hier liegt LLE beim R² vorn, Isomap aber bei Abstandstreue und Trustworthiness.",
}
PRESET_EXPECTED_BANDS = {
    "Gekrümmte Fläche: LLE entrollt": {"verdict": "lle_wins", "r2_lle": (0.92, 1.0), "r2_iso": (0.94, 1.0), "r2_pca": (0.4, 0.6), "fid_lle": (0.8, 0.95), "fid_iso": (0.9, 1.0)},
    "Regularisierung zu schwach": {"verdict": "reg_weak", "r2_lle": (0.55, 0.78), "r2_iso": (0.94, 1.0)},
    "Ohne Regularisierung: nicht definiert": {"verdict": "undefined", "r2_iso": (0.9, 1.0)},
    "Zu kleines k": {"verdict": "k_small", "r2_lle": (0.3, 0.6), "r2_iso": (0.9, 1.0), "fid_lle": (0.2, 0.5)},
    "Rauschen: Isomap ist robuster": {"verdict": "noise", "r2_lle": (0.45, 0.7), "r2_iso": (0.75, 0.92)},
    "Drei Faktoren: dünne Stichprobe": {"verdict": "lle_wins", "r2_lle": (0.45, 0.7), "r2_iso": (0.35, 0.6), "r2_pca": (0.1, 0.3), "fid_iso": (0.6, 0.8)},
}
