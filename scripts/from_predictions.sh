set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" src/headline.py
"$PY" src/analyse.py
"$PY" src/calibration.py
"$PY" src/seed_contrast.py
"$PY" src/backbones.py
"$PY" src/fig_pipeline.py
"$PY" src/fig_construction.py
"$PY" src/fig_backbone.py
"$PY" src/fig_headline.py
"$PY" src/fig_dose.py
