from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import json
import shutil
import os

"""
labels_16_17 = '/Users/moonchaeyeon/PycharmProjects/ml_3/dataset_classification/labels/16_17_labels.json'
labels_18_19 = '/Users/moonchaeyeon/PycharmProjects/ml_3/dataset_classification/labels/18_19_labels.json'
labels_20_21 = '/Users/moonchaeyeon/PycharmProjects/ml_3/dataset_classification/labels/20_21_labels.json'

# 1. open labels.json
with open(labels_20_21, 'r') as f:
    data = json.load(f)

# 2. migrate images to respective folders
for key, info in data.items():
    if len(info["labels"]) == 0:
        print(f'{key} has no labels')
        continue

    label = info['labels'][0]  # "waste" or "no waste"

    original_img_path = f'/Users/moonchaeyeon/Downloads/GEO5017-Project-UrbanWaste/UrbanWaste-images-10k-right/2020_2021/{key}.jpg'    # e.g., "pano_0000...jpg"

    if label == 'no waste':
        folder_name = 'no_waste'
    else:
        folder_name = 'waste'

    # -------------------------------------------------------------
    # (주의) 여기서 실제 이미지가 train, val, test 중 어디에 속해야 하는지
    # 나누는 로직이 추가로 필요합니다. (예: 2016~2019년 사진은 train으로)
    # -------------------------------------------------------------
    target_split = "val"

    # 최종적으로 파일이 도착할 폴더 경로 설정
    dest_folder = f"dataset_classification/{target_split}/{folder_name}"

    # 이미지를 해당 폴더로 복사(Copy)합니다.
    # (원본 파일 경로가 정확히 맞는지 확인 후 실행하세요)
    shutil.copy(original_img_path, os.path.join(dest_folder, os.path.basename(original_img_path)))

print("done")
"""

path = '/Users/moonchaeyeon/PycharmProjects/ml_3/dataset_classification_under_sampling'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print("\n--- Starting Training ---")
model = YOLO("yolo26n-cls.pt")

train_results = model.train(
    data=path,
    epochs=300,
    imgsz=640,
    patience=20,
    device=device,
    project='/Users/moonchaeyeon/PycharmProjects/ml_3/runs/classify',
    name='classification_over_sampling',
)

print("\n--- Evaluating on Validation Set with the Best Model ---")
best_weights_path = Path(train_results.save_dir) / 'weights' / 'best.pt'
model_best = YOLO(str(best_weights_path))
"""
best_weights_path = '/Users/moonchaeyeon/PycharmProjects/ml_3/runs/classify/classification_under_sampling/weights/best.pt'
model_best = YOLO(best_weights_path)
"""
val_results = model_best.val(data=path, split='val')

print("\n--- Best Model Accuracy with Validation Set---")
print(f"  Top-1 Accuracy: {val_results.top1:.4f} ({(val_results.top1 * 100):.2f}%)")

print("\n--- Running Predictions on Test Images ---")
results = model_best.predict(
    source=f'{path}/test_images',
    save=True,
    device='cpu',
    stream=True,
    project='/Users/moonchaeyeon/PycharmProjects/ml_3/runs/classify',
    name='classification_under_sampling_predict'
)

all_predictions = []
waste_cnt = 0
no_waste_cnt = 0

print("start saving predictions...")
for r in results:
    img_name = Path(r.path).name
    top1_idx = r.probs.top1
    conf = float(r.probs.top1conf)
    label = model_best.names[top1_idx]

    if label == 'no_waste':
        no_waste_cnt += 1
    else:
        waste_cnt += 1

    all_predictions.append({
        'img_name': img_name,
        'label': label,
        'conf': conf,
        'box': 'N/A'
    })

print("\nwaste cnt: {}".format(waste_cnt))
print("\nno waste cnt: {}".format(no_waste_cnt))

all_predictions.sort(key=lambda x: x['conf'], reverse=True)
top_100 = all_predictions[:100]

output_file_top100 = '/Users/moonchaeyeon/PycharmProjects/ml_3/runs/classify/top_100_under_sampling_classification.txt'
with open(output_file_top100, 'w', encoding='utf-8') as f:
    f.write("Rank | Image Name | Label | Confidence | Box (xyxy)\n")
    f.write("-" * 60 + "\n")

    for i, pred in enumerate(top_100, 1):
        f.write(f"{i} | {pred['img_name']} | {pred['label']} | {pred['conf']:.4f} | {pred['box']}\n")

output_file_total = '/Users/moonchaeyeon/PycharmProjects/ml_3/runs/classify/total_under_sampling_classification.txt'
with open(output_file_total, 'w', encoding='utf-8') as f:
    f.write("Rank | Image Name | Label | Confidence | Box (xyxy)\n")
    f.write("-" * 60 + "\n")

    for i, pred in enumerate(all_predictions, 1):
        f.write(f"{i} | {pred['img_name']} | {pred['label']} | {pred['conf']:.4f} | {pred['box']}\n")

print(f"\ndone! check '{output_file_total}'")
