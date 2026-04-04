from ultralytics import YOLO

path = '/Users/moonchaeyeon/PycharmProjects/ml_3/data.yaml'

model = YOLO("yolo26n.pt")

# 1. train
model.train(
    data=path,
    epochs=100,
    imgsz=640,
    batch=8,
    patience=10,
    cls=2.0,
    project='/Users/moonchaeyeon/PycharmProjects/ml_3/runs/detect',
    name='urban_waste_v3'
)

# 2. val
model_best = YOLO('/Users/moonchaeyeon/PycharmProjects/ml_3/runs/detect/urban_waste_v2/weights/best.pt')
val_results = model_best.val(data=path)

# 3. predict
results = model_best.predict(
    source='/Users/moonchaeyeon/PycharmProjects/ml_3/images/test',
    save=True,
    conf=0.1
)
