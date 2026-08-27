#!/bin/bash

set -e

BASE_URL="http://127.0.0.1:8001"
IMAGE="/tmp/smoke_test.jpg"

echo "Creating test image..."

python -c "from PIL import Image; Image.new('RGB', (224,224), (255,0,0)).save('$IMAGE')"

echo "Checking health endpoint..."

curl --fail --silent --show-error \
  "$BASE_URL/health"

echo
echo "Health check passed."

echo "Checking prediction endpoint..."

curl --fail --silent --show-error \
  -X POST \
  -F "file=@$IMAGE" \
  "$BASE_URL/predict"

echo
echo "Prediction smoke test passed."

rm -f "$IMAGE"