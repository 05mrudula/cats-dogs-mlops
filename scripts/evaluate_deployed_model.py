import csv
from pathlib import Path

import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


API_URL = "http://127.0.0.1:8001/predict"
DATASET_DIR = Path("data/raw/PetImages")
OUTPUT_FILE = Path("reports/post_deployment_predictions.csv")

results = []

# Collect 5 images from each class
for true_label in ["Cat", "Dog"]:
    image_dir = DATASET_DIR / true_label
    images = sorted(image_dir.glob("*.jpg"))[:5]

    for image_path in images:
        with image_path.open("rb") as image_file:
            response = requests.post(
                API_URL,
                files={"file": (image_path.name, image_file, "image/jpeg")},
                timeout=60,
            )

        response.raise_for_status()
        prediction = response.json()

        results.append({
            "filename": image_path.name,
            "true_label": true_label,
            "predicted_label": prediction["prediction"],
            "confidence": prediction["probability"],
        })


# Calculate performance metrics
y_true = [r["true_label"] for r in results]
y_pred = [r["predicted_label"] for r in results]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="Dog", zero_division=0)
recall = recall_score(y_true, y_pred, pos_label="Dog", zero_division=0)
f1 = f1_score(y_true, y_pred, pos_label="Dog", zero_division=0)


# Save prediction results
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "filename",
            "true_label",
            "predicted_label",
            "confidence",
        ],
    )
    writer.writeheader()
    writer.writerows(results)


print("\nPost-Deployment Model Performance")
print("---------------------------------")
print(f"Requests evaluated : {len(results)}")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1-score            : {f1:.4f}")
print(f"\nResults saved to: {OUTPUT_FILE}")

print("\nIndividual predictions:")
for result in results:
    status = "PASS" if result["true_label"] == result["predicted_label"] else "FAIL"
    print(
        f"{result['filename']:10} "
        f"True={result['true_label']:3} "
        f"Predicted={result['predicted_label']:3} "
        f"Confidence={result['confidence']:.4f} "
        f"[{status}]"
    )
