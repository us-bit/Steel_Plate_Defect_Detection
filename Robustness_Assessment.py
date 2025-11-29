from ultralytics import YOLO
import os
import yaml


def eval_split(model, data_yaml, split_name):
    metrics = model.val(
        data=data_yaml,
        split="test",      # 用 data.yaml 里的 test 字段
        save_json=False,
        plots=False,
        verbose=False,
    )
    print(f"\n===== {split_name} =====")
    print(f"mAP50:     {float(metrics.box.map50):.4f}")
    print(f"mAP50-95:  {float(metrics.box.map):.4f}")
    print(f"Precision: {float(metrics.box.mp):.4f}")
    print(f"Recall:    {float(metrics.box.mr):.4f}")
    return metrics


def ensure_aug_yaml(yaml_path):
    """如果没有增强版 test 的 yaml，就自动生成一份"""
    if os.path.exists(yaml_path):
        return yaml_path

    cfg = {
        "path": r"D:\CV\NEU-DET\NEU-DET",
        "train": "train/images",  # 假的
        "val": "val/images",  # 假的
        "test": r"test_aug_lbl/images",
        "nc": 6,
        "names": [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches",
        ],
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    print(f"[信息] 已生成增强 test 的配置文件: {yaml_path}")
    return yaml_path


def main():
    # 1. 加载你最好的 yolov8s_long 模型
    model_path = r"D:\CV\Steel_Plate_Defect_Detection\runs\train_neu\yolov8s_neu_long\weights\best.pt"
    model = YOLO(model_path)

    # 2. 原始 test 用训练时那个 data.yaml
    data_orig_yaml = r"D:\CV\Steel_Plate_Defect_Detection\data.yaml"

    # 3. 增强 test 写一份专门的 yaml
    data_aug_yaml = ensure_aug_yaml(
        r"D:\CV\Steel_Plate_Defect_Detection\data_test_aug.yaml"
    )

    # 4. 评估 orig test
    m_orig = eval_split(model, data_orig_yaml, "Original Test")

    # 5. 评估 aug test
    m_aug = eval_split(model, data_aug_yaml, "Augmented Test")

    # 6. 做一个简单对比
    print("\n===== 对比结果 (Aug - Orig) =====")
    print(f"ΔmAP50:     {float(m_aug.box.map50 - m_orig.box.map50):.4f}")
    print(f"ΔmAP50-95:  {float(m_aug.box.map - m_orig.box.map):.4f}")
    print(f"ΔPrecision: {float(m_aug.box.mp - m_orig.box.mp):.4f}")
    print(f"ΔRecall:    {float(m_aug.box.mr - m_orig.box.mr):.4f}")


if __name__ == "__main__":
    main()
