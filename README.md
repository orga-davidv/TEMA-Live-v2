# TEMA-Live-v2

Modular TEMA pipeline with optional legacy fallback runner.

## Quickstart

```bash
python -m pip install -U pip
pip install -e . pytest
python run_pipeline.py --run-id smoke
```

## Output structure

Runs write to `outputs/<run_id>/` (or `--out-root <path>/<run_id>/`):

```text
outputs/<run_id>/
  manifest.json
  performance.json
  final_weights.json
  returns_csv_info.json
  ...
```

`manifest.json` includes `schema_version`, `run_id`, `timestamp`, and declared `artifacts`.

## Key flags matrix

| Flag | Default | Purpose |
|---|---:|---|
| `--run-id <id>` | `manual-run` | Output folder name under `outputs/`. |
| `--out-root <dir>` | `outputs` | Change output root directory. |
| `--legacy` | off | Use legacy path (executes only with `TEMA_RUN_LEGACY_EXECUTE=1`). |
| `--stress-enabled` | off | Add stress-scenario artifact generation. |
| `--modular-data-signals` | off | Force modular data/signal pipeline path. |
| `--modular-portfolio` | off | Force modular portfolio allocation path. |
| `--ml-disabled` | off | Disable ML scaling path. |
| `--ml-modular-path` | off | Enable modular ML probability/scalar path. |
| `--template-default-universe` | off | Template-like data/signal defaults. |
| `--no-default-validation-suite` | off | Skip default WF/OOS/bootstrap/MC validation bundle. |
| `--no-validation-graphs` | off | Disable validation PNG chart outputs. |

## ⚠️ Disclaimer

This software is provided for **educational and research purposes only**.
It does not constitute financial advice, investment advice, trading advice,
or any other form of professional advice. The authors and contributors make
no representations or warranties of any kind, express or implied, regarding
the accuracy, completeness, reliability, or suitability of this software
for any purpose.

**Use at your own risk.** Trading financial instruments involves substantial
risk of loss and is not suitable for all investors. Past performance,
backtested results, or simulated returns are not indicative of future results.
Any strategy implemented using this framework may result in partial or total
loss of capital.

The authors accept no liability for any financial losses, damages, or other
consequences arising directly or indirectly from the use of this software.
This project is not affiliated with, endorsed by, or connected to any
financial institution, broker, or regulatory authority.

Always consult a qualified financial advisor before making any investment
decisions. Ensure compliance with all applicable laws and regulations in
your jurisdiction before deploying any trading system.

## ⚠️ Haftungsausschluss

Diese Software wird ausschließlich zu **Bildungs- und Forschungszwecken**
bereitgestellt. Sie stellt keine Finanz-, Anlage-, Handels- oder sonstige
professionelle Beratung dar. Die Autoren und Mitwirkenden übernehmen
keinerlei Gewähr – weder ausdrücklich noch stillschweigend – für die
Richtigkeit, Vollständigkeit, Zuverlässigkeit oder Eignung dieser Software
für einen bestimmten Zweck.

**Nutzung auf eigene Gefahr.** Der Handel mit Finanzinstrumenten ist mit
einem erheblichen Verlustrisiko verbunden und ist nicht für jeden Anleger
geeignet. Vergangene Wertentwicklungen, Backtesting-Ergebnisse oder
simulierte Renditen sind kein verlässlicher Indikator für zukünftige
Ergebnisse. Jede mit diesem Framework umgesetzte Strategie kann zu einem
teilweisen oder vollständigen Kapitalverlust führen.

Die Autoren übernehmen keine Haftung für finanzielle Verluste, Schäden oder
sonstige Folgen, die direkt oder indirekt aus der Nutzung dieser Software
entstehen. Dieses Projekt steht in keiner Verbindung zu einem Finanzinstitut,
Broker oder einer Aufsichtsbehörde und wird von diesen weder unterstützt
noch autorisiert.

Bitte konsultiere einen qualifizierten Finanzberater, bevor du
Anlageentscheidungen triffst. Stelle sicher, dass du alle geltenden
Gesetze und Vorschriften in deinem Land einhältst, bevor du ein
Handelssystem in Betrieb nimmst.
