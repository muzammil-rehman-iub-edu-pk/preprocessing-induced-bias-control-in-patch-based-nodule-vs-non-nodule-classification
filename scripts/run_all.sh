set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/prepare_data.sh
bash scripts/train_all.sh
bash scripts/train_backbones.sh
bash scripts/analyse.sh
bash scripts/figures.sh
