set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" src/fig_overview.py
"$PY" src/fig_pipeline.py
"$PY" src/fig_phases.py
"$PY" src/fig_construction_chain.py
"$PY" src/fig_augment.py
"$PY" src/fig_construction.py
"$PY" src/fig_backbone.py
"$PY" src/fig_headline.py
"$PY" src/fig_dose.py
