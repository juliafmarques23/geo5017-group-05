from ultralytics import YOLO
import os
import torch

def top_100(results, model):
    """A function to find and write the top 100 detections for the leaderboard."""
    all_detections = []

    for result in results:
        img_name = os.path.basename(result.path)
        for box in result.boxes:
            conf = float(box.conf[0])  # confidence score
            cls = int(box.cls[0])  # class ID
            label = model.names[cls]  # class name (px garbage bag)
            all_detections.append({
                'image': img_name,
                'confidence': conf,
                'label': label,
                'box': box.xyxy[0].tolist()
            })

    #sort by confidence (highest first)
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)

    top_100 = all_detections[:100]

    # save .txt file
    with open('top_100_detections_v2.txt', 'w') as f:
        f.write("Rank | Image Name | Label | Confidence | Box (xyxy)\n")
        f.write("-" * 60 + "\n")
        for i, det in enumerate(top_100, 1):
            f.write(f"{i} | {det['image']} | {det['label']} | {det['confidence']:.4f} | {det['box']}\n")

    print(f"✅ :D XD Success! Top 100 detections saved to 'top_100_detections.txt'.")
    print(f"The highest confidence was: {top_100[0]['confidence']:.4f}")


def main():
    # load the model
    model = YOLO("yolo26n.pt")

    # train the model
    results_train = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=15,
        workers=4,
        cls=2.0,
        device=0,
        project='./results',
        name="urban_waste_v2"
    )

    # the path of best model
    best_model_path = os.path.join(results_train.save_dir, 'weights', 'best.pt')
    model = YOLO(best_model_path)

    metrics = model.val()
    print(f"Final Validation mAP50: {metrics.box.map50}")

    results = model.predict(
        source='./datasets/test/images',
        save=True,
        save_txt=True,
        conf=0.1
    )

    top_100(results, model)

    print("done!!")
    print(f"GPU Active: {torch.cuda.get_device_name(0)}")
    print("Check ./results/urban_waste_v2/ for Charts.")


if __name__ == "__main__":
    main()