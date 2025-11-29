import os
from ultralytics import YOLO
import csv
import glob
from datetime import datetime

def main():

    model_path = r"D:\CV\Steel_Plate_Defect_Detection\runs\train_neu\yolov8s_neu_long\weights\best.pt"
    test_path = r"D:\CV\NEU-DET\NEU-DET\test\images"

    # 用 F1 最优阈值
    CONF_TH = 0.586


    base_dir = r"D:\CV\Steel_Plate_Defect_Detection\results"
    os.makedirs(base_dir, exist_ok=True)

    # 自动生成子目录：比如 "pred_s_0586_20240224_235959"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"pred_s_{int(CONF_TH*1000):04d}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    images_save_dir = os.path.join(save_dir, "images")
    os.makedirs(images_save_dir, exist_ok=True)

    # ================================
    # 加载模型
    # ================================
    model = YOLO(model_path)

    # ================================
    # 找到全部测试图像
    # ================================
    imgs = glob.glob(os.path.join(test_path, "*.jpg"))

    # ================================
    # 创建 CSV
    # ================================
    csv_path = os.path.join(save_dir, "pred_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "x1", "y1", "x2", "y2", "class", "confidence"])

        # ================================
        # 对每张图片进行推理
        # ================================
        for img_path in imgs:
            img_name = os.path.basename(img_path)

            results = model.predict(
                img_path,
                save=True,
                project=images_save_dir,
                name="",  # 不创建子目录，直接输出到 images_save_dir
                exist_ok=True,
                conf=CONF_TH,
                verbose=False,
            )

            # ================================
            # 写入 CSV
            # ================================
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    writer.writerow([img_name] + xyxy + [cls, conf])

    print("\n================ 预测完成 ================")
    print(f"预测图片保存目录：{images_save_dir}")
    print(f"CSV 文件保存路径：{csv_path}")
    print("===========================================")


if __name__ == "__main__":
    main()
