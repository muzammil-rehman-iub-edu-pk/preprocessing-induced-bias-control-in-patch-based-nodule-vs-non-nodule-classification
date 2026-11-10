set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
COOLDOWN="${COOLDOWN:-90}"
export NODULE_BIAS_RESULTS="${NODULE_BIAS_RESULTS:-results}/backbones"
EXTRA=()
if [ "${FORCE:-0}" = "1" ]; then
  EXTRA+=(--force)
fi

train() {
  local arm="$1" backbone="$2" res="$3" split="$4" fold="$5" seed="$6"
  local tag="${arm}_${backbone}_${res}_${split}"
  if [ "$fold" != "0" ]; then
    tag="${tag}_fold${fold}"
  fi
  if [ "$seed" != "42" ]; then
    tag="${tag}_seed${seed}"
  fi
  tag="${tag}_clip1"
  if [ -f "${NODULE_BIAS_RESULTS}/runs/${tag}_predictions.csv" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== ${tag}: predictions already present, skipping"
    return
  fi
  echo "=== $(date +%H:%M:%S) ${tag}"
  "$PY" src/train.py --arm "$arm" --backbone "$backbone" --res "$res" --split "$split" \
      --fold "$fold" --seed "$seed" --clipnorm 1.0 ${EXTRA[@]+"${EXTRA[@]}"} || echo "=== ${tag} failed"
  sleep "$COOLDOWN"
}

for res in 128 224; do
  for seed in 42 1234 2026; do
    for backbone in B0 B3 B4 VGG16 R50 R101 D121 IV3; do
      train G4 "$backbone" "$res" single 0 "$seed"
      if [ "$seed" != "42" ] || { [ "$backbone" != "B0" ] && [ "$backbone" != "B3" ]; }; then
        train G2 "$backbone" "$res" single 0 "$seed"
      fi
    done
  done
done

for fold in 1 2 3 4 5; do
  for backbone in B0 VGG16 R50; do
    train G4 "$backbone" 224 kfold "$fold" 42
  done
done
echo "=== backbone training complete"
