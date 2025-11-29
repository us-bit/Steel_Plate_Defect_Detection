import os
from ultralytics import YOLO
import csv
import glob


def main():
    model_path = r"D:\CV\Steel_Plate_Defect_Detection\runs\train_neu\yolov8s_neu_long\weights\best.pt"
    test_path = r"D:\CV\NEU-DET\NEU-DET\test_aug"  # 注意这里是增强后的目录
    save_dir = r"D:\CV\Steel_Plate_Defect_Detection\results\pred_v8s_aug_0586"

    os.makedirs(save_dir, exist_ok=True)

    model = YOLO(model_path)
    imgs = glob.glob(os.path.join(test_path, "*.jpg"))

    csv_path = os.path.join(save_dir, "pred_results_v8s_aug_0586.csv")
    images_subdir = os.path.join(save_dir, "images_0586")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "x1", "y1", "x2", "y2", "class", "confidence"])

        for img_path in imgs:
            img_name = os.path.basename(img_path)

            results = model.predict(
                img_path,
                save=True,
                project=save_dir,
                name="images_0586",
                exist_ok=True,
                conf=0.586,  # 来自 F1 曲线最优点
            )

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    writer.writerow([img_name] + xyxy + [cls, conf])

    print(f"预测完成！结果图在：{images_subdir}")
    print(f"CSV 文件在：{csv_path}")


if __name__ == "__main__":
    main()
