#!/usr/bin/env bash
set -euo pipefail

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"

PROJECT="hemanth-hm"
INSTANCE="xkcd-artist-l4-trainer"
IMAGE_PROJECT="deeplearning-platform-release"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ACTIVE_ZONE=""

cleanup() {
  if [[ -n "$ACTIVE_ZONE" ]]; then
    echo "[GCP-L4] Tearing down GPU VM $INSTANCE in $ACTIVE_ZONE..."
    gcloud compute instances delete "$INSTANCE" \
      --project="$PROJECT" \
      --zone="$ACTIVE_ZONE" \
      --quiet || true
    echo "[GCP-L4] Confirmed teardown of $INSTANCE."
  fi
}
trap cleanup EXIT

for ZONE in us-central1-a us-central1-b us-central1-c us-east1-d us-west1-a; do
  echo "[GCP-L4] Attempting to create g2-standard-4 (NVIDIA L4 24GB GPU) in $ZONE..."
  if gcloud compute instances create "$INSTANCE" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --machine-type="g2-standard-4" \
      --accelerator="type=nvidia-l4,count=1" \
      --maintenance-policy="TERMINATE" \
      --image-project="$IMAGE_PROJECT" \
      --image-family="$IMAGE_FAMILY" \
      --boot-disk-size="100GB" \
      --boot-disk-type="pd-ssd"; then
    ACTIVE_ZONE="$ZONE"
    echo "[GCP-L4] Provisioned $INSTANCE in $ACTIVE_ZONE!"
    break
  fi
done

if [[ -z "$ACTIVE_ZONE" ]]; then
  echo "[GCP-L4] Failed to provision g2-standard-4 across candidate zones."
  exit 1
fi

echo "[GCP-L4] Waiting for SSH readiness on $INSTANCE ($ACTIVE_ZONE)..."
for i in {1..30}; do
  if gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ACTIVE_ZONE" --command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null; then
    break
  fi
  sleep 6
done

echo "[GCP-L4] Packaging training scripts and 226 speech-bubble-masked XKCD panels..."
COPYFILE_DISABLE=1 tar -czf /tmp/xkcd_artist_bundle.tar.gz -C "$ROOT_DIR" training data

gcloud compute scp /tmp/xkcd_artist_bundle.tar.gz "$INSTANCE:/tmp/xkcd_artist_bundle.tar.gz" \
  --project="$PROJECT" --zone="$ACTIVE_ZONE"

echo "[GCP-L4] Running Diffusion UNet LoRA fine-tuning + 5-Head XkcdArtistForGeneration + ONNX Int8 Quantization on NVIDIA L4 GPU..."
gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ACTIVE_ZONE" --command="
  set -euo pipefail
  mkdir -p ~/xkcd-artist
  tar -xzf /tmp/xkcd_artist_bundle.tar.gz -C ~/xkcd-artist
  cd ~/xkcd-artist
  sudo rm -rf /usr/local/lib/python3.10/dist-packages/torchaudio* /usr/local/lib/python3.10/dist-packages/torchvision* || true
  python3 -m pip install -q diffusers peft accelerate transformers safetensors onnx onnxruntime pillow
  python3 training/train_gcp.py
  python3 training/export_onnx.py
  LORA_STEPS=140 python3 training/train_diffusion_lora.py
  tar -czf /tmp/xkcd_artist_artifacts.tar.gz artifacts docs/model_quantized.onnx
"

echo "[GCP-L4] Downloading trained LoRA & ONNX artifacts from $INSTANCE..."
gcloud compute scp "$INSTANCE:/tmp/xkcd_artist_artifacts.tar.gz" /tmp/xkcd_artist_artifacts.tar.gz \
  --project="$PROJECT" --zone="$ACTIVE_ZONE"

tar -xzf /tmp/xkcd_artist_artifacts.tar.gz -C "$ROOT_DIR"
echo "[GCP-L4] Successfully downloaded GCP L4 GPU trained artifacts to $ROOT_DIR/artifacts/xkcd-artist!"
