from ultralytics import YOLO
from pathlib import Path
import cv2

"""HOW TO USE:
1. Setup: Update configuration and local paths.
2. Execution: Run the script to train the model and to predict test dataset.
3. Outputs: 
   - A CSV report ranking the Top 100 'waste' images by confidence
   - The Top 100 'waste' images with respective confidence values
"""

# Configuration
BASE_MODEL = "yolo26n-cls.pt"
NAME = "classification_final_sub_with_under"

# Local paths (Update these for your environment)
DATA_PATH = Path('/Users/moonchaeyeon/PycharmProjects/ml_3/dataset_classification_under')
OUTPUT_PATH = Path('/Users/moonchaeyeon/PycharmProjects/ml_3/runs/classify')


# 1. train the model
print("Starting training...")

model = YOLO(BASE_MODEL)
train_results = model.train(
    data=DATA_PATH,
    epochs=300,
    patience=20,
    imgsz=640,
    device='mps',
    project=OUTPUT_PATH,
    name=NAME
)

print("\nDone training!")


# 2. evaluate the model's performance
print("\nStarting evaluation with test set...")

best_model_path = Path(train_results.save_dir) / 'weights' / 'best.pt'
best_model = YOLO(best_model_path)

test_results = best_model.val(
    data=DATA_PATH,
    split='test',
    device='mps'
)

print(f"\n> Test set Top-1 accuracy: {test_results.top1 * 100:.2f}%")


# 3. predict
print("\nStarting prediction...")

test_dir = DATA_PATH / 'test'
test_image_paths = [
    str(p) for p in test_dir.rglob('*.*')
    if p.suffix.lower() == '.jpg'
]

print(f"> total {len(test_image_paths)} test images")

pred_img_save_dir = Path(OUTPUT_PATH / NAME / 'top_100_pred_images')
pred_img_save_dir.mkdir(parents=True, exist_ok=True)

all_predictions = []
waste_cnt = 0
no_waste_cnt = 0

for i, img_path in enumerate(test_image_paths, 1):
    if i % 100 == 0:
        print(f"Processed {i} / {len(test_image_paths)} images...")

    results = best_model.predict(
        source=img_path,
        save=False,
        device='mps',
        verbose=False
    )

    r = results[0]

    img_name = Path(r.path).name
    true_label = Path(r.path).parent.name
    top1_idx = r.probs.top1
    conf = float(r.probs.top1conf)
    label = best_model.names[top1_idx]

    if label == 'no_waste':
        no_waste_cnt += 1
    else:
        waste_cnt += 1

    all_predictions.append({
        'img_name': img_name,
        'pred_label': label,
        'true_label': true_label,
        'conf': conf,
        'results_object': r  # to save an image later
    })

print(f"\n> Prediction result: total waste count -> {waste_cnt} | total no_waste count -> {no_waste_cnt}")

# 4. Creat top 100 list and save images
all_predictions.sort(key=lambda x: x['conf'], reverse=True)
top_100_waste = [p for p in all_predictions if p['pred_label'] == 'waste'][:100]

print("\nStart saving Top 100 'waste' images...")
for i, pred in enumerate(top_100_waste, 1):
    if i % 20 == 0:
        print(f"Saved {i} / {len(top_100_waste)} images...")

    plotted_img = pred['results_object'].plot()

    base_name = Path(pred['img_name']).stem
    safe_img_name = f"{i:03d}_{base_name}.jpg"
    save_path = pred_img_save_dir / safe_img_name

    cv2.imwrite(str(save_path), plotted_img)

print("\nStart saving Top 100 list...")
output_file = Path(OUTPUT_PATH / NAME / 'top_100_pred_list.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Rank | Image Name           | Predicted Label | True Label | Confidence \n")
    f.write("-" * 65 + "\n")
    for i, pred in enumerate(top_100_waste, 1):
        f.write(
            f"{i:4d} | {pred['img_name']:<20} | {pred['pred_label']:<15} | {pred['true_label']:<15} | {pred['conf']:.4f}\n")

print(f"\nall done!")
print(f"Top 100 'waste' images are saved in a '{pred_img_save_dir}'.")
print(f"Top 100 list txt file is saved in a '{output_file}'.")
