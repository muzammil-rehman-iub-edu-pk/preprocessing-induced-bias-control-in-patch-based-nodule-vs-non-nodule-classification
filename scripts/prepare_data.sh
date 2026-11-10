set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" src/census.py
"$PY" src/extract.py
"$PY" src/audit_coverage.py
"$PY" src/consort.py
"$PY" src/cohort_stats.py
"$PY" src/folds.py
