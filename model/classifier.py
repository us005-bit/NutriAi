"""
FitAI - classifier.py
======================
Loads the trained EfficientNet-B2 model and runs food classification inference.
Place this file in: fitai/backend/model/classifier.py
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import timm
import io
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "efficientnet_b2_best.pth"

# ── Image transform (must match Phase 1 eval_transform_b2) ───────────────────
IMG_SIZE = 260
TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ── Confidence threshold ──────────────────────────────────────────────────────
# If top prediction confidence is below this, we flag it as uncertain
# so the frontend can ask user to confirm manually
CONFIDENCE_THRESHOLD = 40.0  # percent


class FoodClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = None
        self._load_model()

    def _load_model(self):
        """Load model from checkpoint. Reads architecture and classes from the saved file."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                f"Make sure efficientnet_b2_best.pth is in fitai/backend/model/"
            )

        print(f"Loading model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)

        # Read metadata saved during Phase 1 training
        self.classes = checkpoint["classes"]
        model_name   = checkpoint.get("model_name", "efficientnet_b2")
        num_classes  = len(self.classes)

        print(f"  Architecture : {model_name}")
        print(f"  Classes      : {num_classes} → {self.classes}")
        print(f"  Val Accuracy : {checkpoint.get('val_acc', 'N/A')}%")
        print(f"  Device       : {self.device}")

        # Rebuild architecture and load weights
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print("  Model ready ✅")

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        """
        Run inference on image bytes.

        Args:
            image_bytes : raw image bytes from file upload
            top_k       : number of top predictions to return

        Returns:
            {
                "top_prediction" : { "dish": str, "confidence": float },
                "all_predictions": [ { "dish": str, "confidence": float }, ... ],
                "is_uncertain"   : bool   ← True if confidence < threshold
            }
        """
        # Load and preprocess image
        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probs   = F.softmax(outputs, dim=1)[0]

        # Top-k predictions
        top_probs, top_indices = probs.topk(min(top_k, len(self.classes)))

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            predictions.append({
                "dish"      : self.classes[idx],
                "confidence": round(float(prob) * 100, 2)
            })

        top = predictions[0]

        return {
            "top_prediction" : top,
            "all_predictions": predictions,
            "is_uncertain"   : top["confidence"] < CONFIDENCE_THRESHOLD
        }


# ── Singleton — model loads once when backend starts ─────────────────────────
_classifier = None

def get_classifier() -> FoodClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FoodClassifier()
    return _classifier
