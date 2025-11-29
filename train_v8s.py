from ultralytics import YOLO

def main():
    model = YOLO("weights/yolov8s.pt")  # ← v8s，你现在用的

    model.train(
        data="data.yaml",
        imgsz=640,
        epochs=200,                     # ← 50 → 200
        batch=16,
        device=0,
        optimizer="auto",
        project="runs/train_neu",
        name="yolov8s_neu_long",        # ← 不覆盖之前的版本
        val=True,
        workers=0
    )

if __name__ == "__main__":
    main()
