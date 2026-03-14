#!/bin/bash
set -e

pip install -r requirements.txt

echo "Downloading EfficientNet-B2 model weights..."
pip install gdown
gdown "1ugaYQgzkv28hB9TxY8zO2Hct6o2Th7pS" -O model/efficientnet_b2_best.pth
echo "Model downloaded successfully ✅"
