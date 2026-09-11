#!/bin/bash
# One-shot environment setup for Track 3 (InternVL3-8B LoRA eval/reload) on a
# fresh molab sandbox. Consolidates every fix discovered across three
# separate sandbox resets -- run this instead of redoing each step by hand.
#
# Usage: bash scripts/setup_molab_track3.sh
# Assumes: run from inside a freshly-cloned ledger-htr-vt repo, i.e. after
#   git clone https://github.com/IshaanM05/ledger-htr-vt.git && cd ledger-htr-vt
set -e

echo "=== 1. Python deps (unpinned -- InternVL's exact old pins don't build on this Python) ==="
pip3 install --quiet "transformers<5" timm peft safetensors gdown

echo "=== 2. Verify CUDA survived the transformers reinstall ==="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA broke!'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo "=== 3. Data (from the shared Drive folder -- fast path, not gdown --folder) ==="
mkdir -p data/raw
gdown '1Odx_LVVXXt0n4NU64Mld4-aSF3Tg1Y5O' -O /tmp/images.zip
gdown '1QGhkYRYWjQ7Baxso5BzkLkYVHgBF0a9J' -O data/raw/Train.csv
gdown '1MLVEoj-NNXxpl5sMeJxQ4Pra27hL1D3Z' -O data/raw/Test.csv
gdown '14N6hHqPf8h3oKG5V7IbkLNWHATZVz19g' -O data/raw/SampleSubmission.csv
(cd data/raw && unzip -q /tmp/images.zip -d .)
echo "images: $(find data/raw/images -name '*.jpg' | wc -l) (expect 5472)"

echo "=== 4. Fold split (deterministic, seed=42) ==="
PYTHONPATH=src python3 -m ledger_htr.data.cv_split

echo "=== 5. InternVL repo clone + the config-bug patch ==="
if [ ! -d /root/InternVL ]; then
  git clone --depth 1 https://github.com/OpenGVLab/InternVL.git /root/InternVL
fi
python3 scripts/patch_internvl_config_bug.py /root/InternVL

echo "=== 6. Pull the LoRA weights back from Drive ==="
mkdir -p checkpoints/internvl3_8b_lora_fold0
gdown '1XSO3s5JSJTlEMLbkdqxzq0RAMqY_Uljx' -O checkpoints/internvl3_8b_lora_fold0/lora_final.safetensors
gdown '16b_So0535E_8-j99s51Ta_1zqxjrI1aG' -O checkpoints/internvl3_8b_lora_fold0/lora_checkpoint200.safetensors

echo "=== 7. Smoke-test the reload path before trusting it ==="
PYTHONPATH=src python3 -c "
from ledger_htr.extract_lora_adapter import load_lora_weights_into_base_model
model = load_lora_weights_into_base_model('checkpoints/internvl3_8b_lora_fold0/lora_final.safetensors')
print('reload OK:', type(model).__name__)
"

echo "=== Setup complete. Ready for: ==="
echo "PYTHONPATH=src python3 -m ledger_htr.predict_vlm --model-path checkpoints/internvl3_8b_lora_fold0/lora_final.safetensors --mode test --num-beams 4 --out-csv submissions/internvl3_lora_submission_beam4.csv"
echo ""
echo "NOTE: the automatic Drive backup loop (rclone, every 5 min) needs its"
echo "config re-pasted on this fresh sandbox -- that's a per-machine OAuth"
echo "token, not something this script can carry over. Ask for it to be"
echo "re-applied to /home/marimo/.config/rclone/rclone.conf (note: NOT"
echo "/root/.config/rclone/ -- rclone resolves its config path independent"
echo "of \$HOME on this sandbox image) if you want backups running again."
