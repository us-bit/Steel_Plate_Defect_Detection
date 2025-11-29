import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ============ 配置部分 ============
BASE = Path(r"D:\CV\NEU-DET\NEU-DET")

IMG_DIR = BASE / "images_all"   # 临时合并目录
LBL_DIR = BASE / "labels_all"

# 数据集输出路径
train_img = BASE / "train" / "images"
train_lbl = BASE / "train" / "labels"
val_img   = BASE / "val" / "images"
val_lbl   = BASE / "val" / "labels"
test_img  = BASE / "test" / "images"
test_lbl  = BASE / "test" / "labels"

# 清理旧目录
for d in [train_img, train_lbl, val_img, val_lbl, test_img, test_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# ============ 第一步：收集所有图片 ============
def collect_all():
    print("合并 train + valid 所有图片和 labels ...")

    IMG_DIR.mkdir(exist_ok=True)
    LBL_DIR.mkdir(exist_ok=True)

    for part in ["train", "valid"]:
        img_src = BASE / part / "images"
        lbl_src = BASE / part / "labels"

        for img in img_src.glob("*.jpg"):
            shutil.copy(img, IMG_DIR / img.name)

        for lbl in lbl_src.glob("*.txt"):
            shutil.copy(lbl, LBL_DIR / lbl.name)

    print("合并完成。")

# ============ 第二步：按类别分类 ============
def load_by_class():
    print("按类别读取 labels ...")
    class_to_files = defaultdict(list)

    for txt in LBL_DIR.glob("*.txt"):
        with open(txt, "r") as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            cls = int(first_line.split()[0])
            class_to_files[cls].append(txt.stem)

    return class_to_files

# ============ 第三步：划分 8:1:1 ============
def split_dataset(class_to_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    print("开始拆分 8:1:1 ...")
    for cls, files in class_to_files.items():
        random.shuffle(files)
        total = len(files)

        n_train = int(total * train_ratio)
        n_val   = int(total * val_ratio)
        n_test  = total - n_train - n_val

        train_files = files[:n_train]
        val_files   = files[n_train:n_train+n_val]
        test_files  = files[n_train+n_val:]

        move_files(train_files, train_img, train_lbl)
        move_files(val_files, val_img, val_lbl)
        move_files(test_files, test_img, test_lbl)

        print(f"类 {cls}: total={total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

def move_files(stems, img_dir, lbl_dir):
    for name in stems:
        img = IMG_DIR / f"{name}.jpg"
        lbl = LBL_DIR / f"{name}.txt"
        if img.exists():
            shutil.copy(img, img_dir / img.name)
        if lbl.exists():
            shutil.copy(lbl, lbl_dir / lbl.name)

# ============ 主函数 ============
if __name__ == "__main__":
    collect_all()
    class_map = load_by_class()
    split_dataset(class_map)
    print("\n数据集成功重新划分为 8:1:1 ！")
