import csv
import os
from collections import Counter, defaultdict
import numpy as np

# NEU-DET 类别名（按你的 data.yaml）
CLASS_NAMES = ['crazing', 'inclusion', 'patches',
               'pitted_surface', 'rolled-in_scale', 'scratches']


def analyse_csv(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[错误] 找不到 CSV 文件：{csv_path}")
        return

    print(f"[信息] 正在分析：{csv_path}")

    per_image_counts = Counter()
    per_class_counts = Counter()
    per_class_conf = defaultdict(list)
    all_conf = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[警告] CSV 为空，没有任何预测结果。")
        return

    for row in rows:
        img = row["image_id"]
        cls = int(row["class"])
        conf = float(row["confidence"])

        per_image_counts[img] += 1
        per_class_counts[cls] += 1
        per_class_conf[cls].append(conf)
        all_conf.append(conf)

    # 1. 总体情况
    print("\n==== 1. 总体情况 ====")
    print(f"总检测框数量：{len(rows)}")
    print(f"涉及图片数量：{len(per_image_counts)}")
    print(f"所有预测平均置信度：{np.mean(all_conf):.4f}")
    print(f"置信度中位数：{np.median(all_conf):.4f}")
    print(f"置信度范围：[{min(all_conf):.4f}, {max(all_conf):.4f}]")

    # 2. 每类检测数量 & 置信度
    print("\n==== 2. 按类别统计 ====")
    for cls_idx in sorted(per_class_counts.keys()):
        name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
        cnt = per_class_counts[cls_idx]
        confs = np.array(per_class_conf[cls_idx])
        print(f"- 类别 {cls_idx} ({name})")
        print(f"  检测框数量：{cnt}")
        print(f"  置信度：mean={confs.mean():.4f}, median={np.median(confs):.4f}, "
              f"min={confs.min():.4f}, max={confs.max():.4f}")

    # 3. 每张图检测框数量分布
    print("\n==== 3. 每张图片检测数量分布 ====")
    counts = np.array(list(per_image_counts.values()))
    print(f"  平均每图检测框数量：{counts.mean():.3f}")
    print(f"  中位数：{np.median(counts):.3f}")
    print(f"  min={counts.min()}, max={counts.max()}")

    # 找“检测很多”和“检测很少”的图片
    top_k = 5
    print(f"\n  检测框最多的 {top_k} 张图片：")
    for img, c in per_image_counts.most_common(top_k):
        print(f"    {img}: {c} 个框")

    print(f"\n  检测框最少的 {top_k} 张图片：")
    for img, c in sorted(per_image_counts.items(), key=lambda x: x[1])[:top_k]:
        print(f"    {img}: {c} 个框")

    print("\n[完成] CSV 分析结束。")


if __name__ == "__main__":
    # TODO: 换成你当前要分析的 CSV 路径，比如 v8s_long 的预测结果
    csv_path = r"D:\CV\Steel_Plate_Defect_Detection\results\pred_v8s_aug_0586\pred_results_v8s_aug_0586.csv"
    analyse_csv(csv_path)
