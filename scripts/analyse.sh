set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" src/probe.py
"$PY" src/stratify.py
"$PY" src/headline.py
"$PY" src/analyse.py
"$PY" src/calibration.py
"$PY" src/seed_contrast.py
"$PY" src/dose.py
"$PY" src/backbones.py
"$PY" src/dedup_audit.py
