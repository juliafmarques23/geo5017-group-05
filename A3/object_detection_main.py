import os
import csv
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

"""HOW TO USE:
1. Setup: Update 'DATA_YAML' and local paths (TEST_IMAGES_PATH, LABELS_PATH).
2. Execution: Run the script to train for 150 epochs and evaluate the Test set.
3. Outputs: 
   - A CSV report ranking the Top 100 images by confidence
   - 100 images with bounding boxes for the Localization bonus
   - Automated Precision at 100 (P@100) calculation

Ensure data.yaml correctly points to:
   train: ./datasets/train/images
   val: ./datasets/val/images (Used for hyperparameter tuning) **Skip that tuning due to ram issues!
   test: ./datasets/test/images (Used for final Precision@100 evaluation)
   """
# Configuration
DATA_YAML = "data.yaml"
BASE_MODEL = "yolo26n.pt"
PROJECT = './results'
NAME = "urban_waste_v6"

# Local paths (Update these for your environment)
TEST_IMAGES_PATH = r'C:\Users\evang\Documents\Q3\GEO5017\ass03\datasets\test\images'
LABELS_PATH = r'C:\Users\evang\Documents\Q3\GEO5017\ass03\datasets\clean\test_data_2022\labels\train'
OUTPUT_TOP_100_DIR = r'C:\Users\evang\Documents\Q3\GEO5017\ass03\top_100_predictions_v7'
CSV_REPORT_PATH = r'C:\Users\evang\Documents\Q3\GEO5017\ass03\waste_report_v7.csv'


def leaderboard(results, model, true_waste_names):
    """Filters for waste, ranks by confidence, and saves the Top 100 list"""
    prediction_list = []

    print("Ranking unique images by confidence...")
    for r in results:
        full_filename = os.path.basename(r.path)
        name_no_ext = os.path.splitext(full_filename)[0]

        # Only process if boxes exist
        if len(r.boxes) > 0:
            max_idx = torch.argmax(r.boxes.conf)
            max_conf = float(r.boxes.conf[max_idx])
            label = model.names[int(r.boxes.cls[max_idx])]
        else:
            max_conf = 0.0
            label = "none"

        prediction_list.append({
            "filename": full_filename,
            "conf": max_conf,
            "label": label,
            "is_actually_waste": name_no_ext in true_waste_names,
            "result_obj": r
        })

    # Sort descending and take top 100
    prediction_list.sort(key=lambda x: x['conf'], reverse=True)
    top_100 = prediction_list[:100]

    # Save annotated images and CSV
    os.makedirs(OUTPUT_TOP_100_DIR, exist_ok=True)
    correct_count = 0

    with open(CSV_REPORT_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Rank', 'Confidence', 'Label', 'Status', 'Filename'])

        for i, p in enumerate(top_100, 1):
            status = "CORRECT" if p['is_actually_waste'] else "UNLABELED_OR_FP"
            if p['is_actually_waste']: correct_count += 1

            writer.writerow([i, round(p['conf'], 4), p['label'], status, p['filename']])

            # Save image with boxes drawn
            annotated_frame = p['result_obj'].plot()
            dst_name = f"rank{i:03d}_{status}_{p['filename']}"
            cv2.imwrite(os.path.join(OUTPUT_TOP_100_DIR, dst_name), annotated_frame)

    return correct_count


def main():
    #1 initialize model
    model = YOLO(BASE_MODEL)

    #2 Training phase
    print("Phase 1: Final Training on Temporal Split (2016-2019)...")
    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=8,
        workers=0,
        device=0,
        # Handling imbalance via hyperparameter selection
        cls=2.0,
        mosaic=0.5,
        copy_paste=0.1,
        project=PROJECT,
        name=NAME
    )

    #3 Validation on Test set (2022-2023)
    best_weights = Path(PROJECT) / NAME / 'weights' / 'best.pt'
    # best_weights = r'C:\Users\evang\Documents\Q3\GEO5017\ass03\runs\detect\results\urban_waste_v6\weights\best.pt'
    best_model = YOLO(best_weights)

    print("Phase 2: Test Set Evaluation (2022-2023)...")
    best_model.val(data=DATA_YAML, split='test')

    #4 Generate Top 100 leaderboard
    print("Phase 3: Generating Result Visuals and P@100")
    true_waste_names = {os.path.splitext(f)[0] for f in os.listdir(LABELS_PATH) if f.endswith('.txt')}
    test_results = best_model.predict(source=TEST_IMAGES_PATH, conf=0.1, stream=True, iou=0.45, verbose=False)

    #calculate final precision at 100
    precision_count = leaderboard(test_results, best_model, true_waste_names)

    print(f"\nFinal Precision@100: {precision_count}%")

if __name__ == "__main__":
    main()
