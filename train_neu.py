from ultralytics import YOLO

def main():
    # 1. 加载模型（YOLOv8n）
    model = YOLO("weights/yolov8n.pt")


    # 2. 开始训练
    model.train(
        data="D:\CV\Steel_Plate_Defect_Detection\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,        # 显存不够就改成 8
        device=0,        # GPU
        workers=0,       # ★ 关键：Windows 下先设 0，避免多进程坑
        project="runs/train_neu",
        name="yolov8n_neu",
        pretrained=True,
        verbose=True
    )

if __name__ == "__main__":
    main()
