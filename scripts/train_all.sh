set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
COOLDOWN="${COOLDOWN:-90}"
RESULTS_DIR="${NODULE_BIAS_RESULTS:-results}"
EXTRA=()
if [ "${FORCE:-0}" = "1" ]; then
  EXTRA+=(--force)
fi

RUNS=(
  "G0 B0 128 single 0 42"
  "G0 B0 128 single 0 1234"
  "G0 B0 128 single 0 2026"
  "G1 B0 128 single 0 42"
  "G1 B0 128 single 0 1234"
  "G1 B0 128 single 0 2026"
  "G2 B0 128 single 0 42"
  "G2 B0 128 single 0 1234"
  "G2 B0 128 single 0 2026"
  "G3 B0 128 single 0 42"
  "G4 B0 128 single 0 42"
  "G2 B0 96 single 0 42"
  "G2 B0 224 single 0 42"
  "G2 B3 128 single 0 42"
  "G2 B3 224 single 0 42"
  "G2 B3 300 single 0 42"
  "G3 B0 224 kfold 1 42"
  "G3 B0 224 kfold 2 42"
  "G3 B0 224 kfold 3 42"
  "G3 B0 224 kfold 4 42"
  "G3 B0 224 kfold 5 42"
  "D100 B0 128 single 0 42"
  "D366 B0 128 single 0 42"
  "D150 B0 128 single 0 42"
  "D200 B0 128 single 0 42"
)

for spec in "${RUNS[@]}"; do
  read -r arm backbone res split fold seed <<< "$spec"
  tag="${arm}_${backbone}_${res}_${split}"
  if [ "$fold" != "0" ]; then
    tag="${tag}_fold${fold}"
  fi
  if [ "$seed" != "42" ]; then
    tag="${tag}_seed${seed}"
  fi
  if [ -f "${RESULTS_DIR}/runs/${tag}_predictions.csv" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== ${tag}: predictions already present, skipping"
    continue
  fi
  echo "=== $(date +%H:%M:%S) ${tag}"
  "$PY" src/train.py --arm "$arm" --backbone "$backbone" --res "$res" --split "$split" \
      --fold "$fold" --seed "$seed" ${EXTRA[@]+"${EXTRA[@]}"} || echo "=== ${tag} failed"
  sleep "$COOLDOWN"
done
echo "=== training complete"
