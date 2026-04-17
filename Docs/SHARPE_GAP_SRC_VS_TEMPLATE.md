# Sharpe-Gap Analyse — **Template Benchmark** vs **src/** Runs (TEMA-Live-v2)

> **Kurzfazit:** Das Sharpe-Gap kommt **nicht** primär von der Sharpe-Formel, sondern (1) vom **Return-Stream** (Template nutzt *Strategie-Returns*, src nutzt *Buy&Hold pct_change*), (2) von **Train/Test-Splitting/Zeitraum** (Template per Asset, src global auf Panel) und (3) von **Pipeline-Feature-Mismatch** (BL/Views, Grid-Search, Vol-Target, Kostenmodell/Rebalancing).

---

## 1) Ground Truth (Benchmark): Template-Metriken sind reproduzierbar

Template schreibt die Portfolio-Test-Returns in:
- `Template/portfolio_test_returns.csv` (Spalte `portfolio_return`)
- `Template/portfolio_test_returns_ml.csv` (Spalte `portfolio_return_ml`)

Aus diesen CSVs reproduziert man exakt die Benchmarks:

| Quelle | Perioden | annual_return | annual_vol | Sharpe | max_drawdown |
|---|---:|---:|---:|---:|---:|
| **Template strategy portfolio test** (`Template/portfolio_test_returns.csv`) | 2028 | 0.103254 | 0.100500 | **1.027408** | -0.135089 |
| **Template strategy portfolio test (ML overlay)** (`Template/portfolio_test_returns_ml.csv`) | 2028 | 0.108154 | 0.088728 | **1.218938** | -0.125063 |

**Wichtig:** Die Template-Portfolio-Returns sind **Strategie-Returns** (TEMA/EMA-Strategie inkl. Pipeline-Logik), nicht einfach Markt-/Buy&Hold-Returns.

Beleg im Code (Template erzeugt Strategie-Returns pro Asset):
- `Template/TEMA-TEMPLATE(NEW_).py:1717-1793` — `run_asset_pipeline(...)` ruft
  - `build_strategy_returns_for_combo(...)` für `train_returns`/`test_returns`
  - und zusätzlich **Buy&Hold** als Referenz (`train_bh_returns`, `test_bh_returns`) (`pct_change()`).

---

## 2) Beispiel-SRC-Run — **Template-Parity (nach Fix)**

Reproduktion (modular/src, Template ist Benchmark):
```bash
python run_pipeline.py \
  --run-id parity-template \
  --out-root outputs_outroot_test \
  --template-default-universe \
  --modular-data-signals \
  --modular-portfolio \
  --ml-disabled
```

Resultat (aus `outputs_outroot_test/parity-template/performance.json`):

| Quelle | Perioden | annual_return | annual_vol | Sharpe | max_drawdown |
|---|---:|---:|---:|---:|---:|
| **src modular (parity mode)** | 2028 | 0.103254 | 0.100500 | **1.027408** | -0.135089 |

✅ **Parity erreicht** (Sharpe/Return/Vol/Periods matchen den Template-Benchmark innerhalb numerischer Rundungsfehler).

### 2.1) Template-ML Overlay Parity (Sharpe ~1.2189)

Reproduktion (modular/src, Template ist Benchmark):
```bash
python run_pipeline.py \
  --run-id parity-template-ml \
  --out-root outputs_outroot_test \
  --template-default-universe \
  --modular-data-signals \
  --modular-portfolio \
  --ml-template-overlay
```

- Output:
  - `outputs_outroot_test/parity-template-ml/portfolio_test_returns.csv`
  - `outputs_outroot_test/parity-template-ml/portfolio_test_returns_ml.csv`
- Erwartung: `portfolio_test_returns_ml.csv` matcht `Template/portfolio_test_returns_ml.csv` innerhalb numerischer Rundungsfehler (typisch max abs diff ~1e-15) und reproduziert Sharpe **~1.218938**.

Alternativ als Helper:
```bash
python scripts/ml/run_template_ml_overlay.py --run-id parity-template-ml --out-root outputs_outroot_test
```

> Historisch (vor dem Fix) lag die src-Pipeline in Template-Default-Universe Mode deutlich daneben (u.a. falscher Return-Stream / falsches Split-/Alignment) und kam nur auf ~1124 Perioden im Testfenster.

---

## 3) Harte Evidenz: Wenn man Template-BL-Weights auf src Buy&Hold-Returns anwendet, bleibt Sharpe niedrig

Template exportiert BL-Weights:
- `Template/black_litterman_weights.csv`

Wenn man damit (oder ähnlich guten Gewichten) **Buy&Hold pct_change()** über denselben Test-Zeitraum bewertet:

| Return-Stream | Fenster | Sharpe |
|---|---|---:|
| **Buy&Hold** (aus `merged_d1` pct_change) + Template BL Weights | Template-Test-Index (2028 Tage) | **0.837977** |
| **Template Strategie-Returns** (Portfolio) | Template-Test-Index (2028 Tage) | **1.027408** |

Zusätzlicher Hinweis: Korrelation Template-Strategie-Returns vs Buy&Hold-Portfolio-Returns ist nur moderat:
- `corr ≈ 0.4063`

**Interpretation:** Selbst „richtige“/Template-ähnliche Gewichte heben Buy&Hold nicht auf Template-Sharpe. Der Haupttreiber ist der **Return-Stream** (Strategie vs Buy&Hold) + Zeitraum/Splitting.

---

## 4) Was ist **NICHT** der Bug? (Sharpe-Formel ist kompatibel)

Template definiert Sharpe als:
- `annualized_return / annualized_volatility` mit geometrischer Annualisierung und `std(ddof=0)`.
  - `Template/TEMA-TEMPLATE(NEW_).py:999-1002`

src verwendet in `compute_backtest_metrics` exakt dieselbe Sharpe-Semantik:
- `src/tema/backtest.py:118-125` (annual_return geom / annual_vol std*sqrt(ann) / sharpe = annual_return/annual_vol)

-> **Sharpe-Formel ist nicht der Auslöser** des riesigen Gaps.

---

# 5) **Fehler/Abweichungen im src/**, die das Sharpe-Gap auslösen

Im Folgenden sind die *src-seitigen* Ursachen (gegen Template als Benchmark) in „Impact“-Reihenfolge.

---

## A) src backtestet den falschen Return-Stream (**Buy&Hold statt Strategie-Returns**)  ✅ Hauptursache

### Template (Benchmark)
- Pro Asset wird eine **Strategie** gebaut (Grid-Search + Signals + Trades + Kosten) und daraus entstehen `train_returns`/`test_returns`.
  - `Template/TEMA-TEMPLATE(NEW_).py:1758-1761`.
- Portfolio wird auf `test_returns_df` (Strategie-Matrix) evaluiert:
  - `Template/TEMA-TEMPLATE(NEW_).py:1917-1927`.

### src (vor Fix / Root-Cause)
- Backtest-Stage nutzte im Template-Default-Universe Pfad **Buy&Hold** (`pct_change`) statt Strategie-Returns.

**Warum macht das Sharpe kaputt?**
- Buy&Hold-Returns haben in diesem Datensatz (bei Template-Weights) Sharpe ~0.84 (Template-Testfenster), während Template-Strategie-Returns Sharpe ~1.03 liefern.
- Solange src nicht dieselben Strategie-Returns generiert, kann src den Benchmark **nicht** erreichen.

**Fix (jetzt umgesetzt für Template-Parity):**
- In `template_default_universe` wird eine **Strategie-Return-Matrix** erzeugt (`train_strategy_returns`/`test_strategy_returns`, inkl. fee+slippage) und der Backtest nutzt diese Returns.
- Zusätzlich werden für strikte Parity die Template-Artefakte (`Template/asset_strategy_summary.csv`, `Template/black_litterman_weights.csv`) genutzt, so dass Gewichte/Combos mit dem Benchmark übereinstimmen.

---

## B) Train/Test-Splitting ist nicht Template-parallel (globales Panel-Splitting statt per Asset)  ✅ großer Treiber

### Template
- Split passiert **per Asset-Serie**:
  - `Template/TEMA-TEMPLATE(NEW_).py:1730` → `split_train_test(close, cfg.train_ratio)`.

### src
- Split passiert **einmal** auf dem gesamten Panel:
  - `src/tema/data/splitter.py:6-26`.
  - Wird im Runner direkt so genutzt: `src/tema/pipeline/runner.py:56`.

**Impact:**
- src Test-Fenster in Template-Universe Mode hat z.B. **1124 Perioden**, Template-Test-Serie **2028**.
- Alle subsequent Steps (Signals, Alphas, Gewichte, Vol-Proxy, Metrics) laufen auf einem anderen Zeitraum.

**Fix später:** Per-Asset Split (oder eine Äquivalenz-Regel) + dann erst „union align + fillna(0)“ wie Template.

---

## C) Template-Default-Universe „Mode“ in src ist unvollständig: Portfolio-Stage ≠ Template BL/Views  ✅ struktureller Mismatch

### Template
- `view_q` kommt aus **annualisierten Strategie-Train-Returns**:
  - `Template/TEMA-TEMPLATE(NEW_).py:1786-1787`.
- BL-Weights werden aus `train_returns_df` (Strategie) + `view_q` gebaut:
  - `Template/TEMA-TEMPLATE(NEW_).py:902-975`.

### src
- `expected_alphas` werden aus **letztem Signal × letzte Return** gebaut:
  - `src/tema/pipeline/runner.py:243-246`.
- In `template_default_universe` wird `portfolio_modular_enabled` effektiv ausgeschaltet:
  - `use_modular_portfolio = bool(cfg.portfolio_modular_enabled and not cfg.template_default_universe)`
  - `src/tema/pipeline/runner.py:256-257`.
  - Ergebnis ist dann „legacy-signal-normalization“ als Gewichtung (`src/tema/pipeline/runner.py:275-283`).

**Impact:**
- Selbst wenn BL-Weights vs Signal-Normalization hier nicht der #1-Treiber ist (Buy&Hold dominiert), ist es für echte Parity ein „Fehler“ gegenüber Benchmark.

**Fix später:** In Template-Mode BL nicht deaktivieren, sondern explizit Template-BL (oder eine mathematisch äquivalente Implementierung) verwenden.

---

## D) Vol-Target Semantik ist in src anders (und standardmäßig wirkungslos für BL)  ⚠️ Metrics mismatch (vol/return), Sharpe meist invariant

### Template
- Vol-Target wird **immer** (wenn enabled) auf BL Train/Test Returns angewandt:
  - `Template/TEMA-TEMPLATE(NEW_).py:1014-1053` (enabled check, ref selection, scalar, apply).

### src
- Vol-Target greift nur in `_scaling_stage`, aber nur wenn `vol_target_apply_to_ml` True:
  - `src/tema/pipeline/runner.py:392-400`.

**Beobachtung:**
- Ohne Vol-Target hat src `annual_vol` typischerweise viel kleiner als Template (Template zielt ~0.10).
- Mit `--vol-target-apply-to-ml` kann src annual_vol nahe 0.10 bringen, Sharpe bleibt ~gleich (wie erwartet).

**Fix später:** Vol-Target sollte (Template-like) als Return-Scalar auf BL-Returns laufen (und optional auch ML), nicht nur als Feature-Flag über `apply_to_ml`.

---

## E) (Risiko für „riesiges Sharpe-Gap“ in anderen Runs) tägliche Signal-Gewichtspfad-Backtests + Turnover-Kosten  ⚠️ kann Sharpe massiv drücken

### src Verhalten
- Wenn `modular_data_signals_enabled` **und** `backtest_static_weights_in_template` **false**, dann:
  - signal_df wird generiert (`runner.py:164-172`)
  - daraus daily weights (`build_weight_schedule_from_signals`, `runner.py:173-174`)
  - und dann Simulation mit Turnover-Kosten:
    - `src/tema/backtest.py:74-86` (turnover = sum(|cur-prev|), pnl = dot(cur, ret[t]), costs = turnover*(fee+slippage)).

### Template Verhalten
- Portfolio-Evaluation in Template ist in diesem Schritt **statisch** (BL weights) und ohne Portfolio-Turnover-Kosten (Kosten stecken in Strategie-Returns pro Asset).

**Impact:**
- Sobald src in diesen „daily reweight + costs“-Pfad fällt, kann die Performance dramatisch einbrechen (je nach Turnover). Genau das erklärt die sehr niedrigen Sharpe-Werte in manchen `outputs/*` Artefakten (hohe annualized_turnover).

---

## F) Execution-Semantik: src verwendet „previous weights“ pro Period (walk-forward execution)  ⚠️ kann Metriken verschieben

- `src/tema/backtest.py:74-76` setzt `executed = [w[0], w[:-1]]`.
- Template `evaluate_weighted_portfolio` multipliziert Returns mit Gewichten „same time index“ (`Template/TEMA-TEMPLATE(NEW_).py:981-983`).

Für **statische** Gewichte egal, für **dynamische** Gewichte jedoch ein weiterer Parity-Delta.

---

## G) Template hat Asset-spezifische Grid-Search/Validation; src nutzt feste Parameter  ✅ Qualitäts-/Alpha-Treiber

Template wählt pro Asset die beste (ema1,ema2,ema3)-Kombi via Validation:
- `Template/TEMA-TEMPLATE(NEW_).py:1750-1757` (`choose_best_combo_with_validation`).

src nutzt feste fast/slow Perioden und einen sehr simplen `sign(fast-slow)` Signal:
- `src/tema/signals/tema.py:21-52`.

**Impact:**
- Template extrahiert deutlich mehr „Alpha“ (im Sinne von Strategie-Return-Stream) als ein fixed-parameter crossover.

---

## H) Parity-Metrics-Bridge kann die Metriken „faken“ (Maskierung statt Fix)  ⚠️ Debug/Reporting-Falle

- `run_pipeline.py:60-103` überschreibt `performance.json` mit den Legacy-CSV-Metriken (`Template/bl_portfolio_metrics.csv`).

Das ist nützlich für Vergleich/CI, aber **kein Fix** und kann die Ursachen verschleiern.

---

# 6) „Done“-Kriterium für echte Parity (für die spätere Fix-Phase)

Wenn Template der Benchmark ist, dann muss src (mindestens) diese Punkte matchen:
1. **Strategie-Returns pro Asset** (wie Template `build_strategy_returns_for_combo`) statt Buy&Hold `pct_change`.
2. **Per-Asset Split** (`split_train_test` auf jeder Serie) + dann erst Align/Fill wie Template.
3. **view_q / BL-Weights** aus Strategie-Train-Returns (annualize), nicht aus `latest_signal * latest_ret`.
4. **Vol-Target** auf BL-Returns (Return-Scalar) wie Template.
5. Kostenmodell konsistent: keine doppelte/inkonsistente Kostenbelastung (Strategy Costs vs Portfolio Turnover Costs).

---

## Appendix: Quick Links (Code)

- Template Strategie-Pipeline: `Template/TEMA-TEMPLATE(NEW_).py:1717-1793`
- Template Portfolio-Assembly: `Template/TEMA-TEMPLATE(NEW_).py:1917-1927`
- Template BL Weights: `Template/TEMA-TEMPLATE(NEW_).py:902-975`
- Template Metrics + Vol Target: `Template/TEMA-TEMPLATE(NEW_).py:977-1053`
- src Backtest Stage (pct_change Returns): `src/tema/pipeline/runner.py:136-196`
- src Portfolio Stage (latest signal * return): `src/tema/pipeline/runner.py:229-285`
- src Splitter (global panel split): `src/tema/data/splitter.py:6-26`
- src Backtest Execution/Costs: `src/tema/backtest.py:55-90`
- Parity Metrics Bridge: `run_pipeline.py:60-103`
