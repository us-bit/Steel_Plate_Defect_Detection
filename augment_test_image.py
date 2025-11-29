import os
import cv2
import glob
import numpy as np


# ================== 路径配置 ==================
SRC_IMG_DIR = r"D:\CV\NEU-DET\NEU-DET\test\images"
SRC_LABEL_DIR = r"D:\CV\NEU-DET\NEU-DET\test\labels"

# 新的数据集（带标签的增强 test）
DST_IMG_DIR = r"D:\CV\NEU-DET\NEU-DET\test_aug_lbl\images"
DST_LABEL_DIR = r"D:\CV\NEU-DET\NEU-DET\test_aug_lbl\labels"

os.makedirs(DST_IMG_DIR, exist_ok=True)
os.makedirs(DST_LABEL_DIR, exist_ok=True)


# ================== 图像增强函数 ==================
def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """对比度 + 亮度"""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_gaussian_noise(img, mean=0, std=10):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def rotate_image_and_matrix(img, angle=10):
    """返回 旋转后的图像 + 仿射矩阵 M"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return rotated, M


def shift_image_and_matrix(img, dx=5, dy=5):
    """平移图像 + 返回平移矩阵 M"""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return shifted, M


# ================== 标签相关函数 ==================
def load_yolo_labels(label_path):
    """
    读取 YOLO 标签:
    格式: cls x_center y_center w h (全部是归一化到 [0,1] 的)
    返回: [(cls, xc, yc, w, h), ...]
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            boxes.append((cls, xc, yc, w, h))
    return boxes


def save_yolo_labels(label_path, boxes):
    """
    保存 YOLO 标签:
    boxes: [(cls, xc, yc, w, h), ...] 归一化形式
    """
    if not boxes:
        # 可以选择保留空文件，也可以直接不写；这里保留空文件
        open(label_path, "w", encoding="utf-8").close()
        return

    with open(label_path, "w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_corners(box, img_w, img_h):
    """YOLO (xc,yc,w,h) → 像素坐标四个角点"""
    cls, xc, yc, w, h = box
    xc *= img_w
    yc *= img_h
    bw = w * img_w
    bh = h * img_h
    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2
    # 四个角点
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)
    return cls, corners


def corners_to_yolo(cls, corners, img_w, img_h):
    """四个角点 → YOLO 归一化 (xc,yc,w,h)，并裁剪到图像范围"""
    xs = corners[:, 0]
    ys = corners[:, 1]

    x1 = np.clip(xs.min(), 0, img_w - 1)
    y1 = np.clip(ys.min(), 0, img_h - 1)
    x2 = np.clip(xs.max(), 0, img_w - 1)
    y2 = np.clip(ys.max(), 0, img_h - 1)

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1e-3 or bh <= 1e-3:
        return None  # 这个框太小/无效，直接丢掉

    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0

    # 归一化
    xc /= img_w
    yc /= img_h
    bw /= img_w
    bh /= img_h

    # 再做一层安全裁剪
    xc = np.clip(xc, 0.0, 1.0)
    yc = np.clip(yc, 0.0, 1.0)
    bw = np.clip(bw, 0.0, 1.0)
    bh = np.clip(bh, 0.0, 1.0)

    return (cls, float(xc), float(yc), float(bw), float(bh))


def apply_affine_to_box(box, M, img_w, img_h):
    """
    对一个 YOLO box 应用仿射变换 M (2x3)
    返回新的 YOLO box (cls, xc,yc,w,h) 或 None
    """
    cls, corners = yolo_to_corners(box, img_w, img_h)
    # 将 corners 扩展成齐次坐标 [x, y, 1]
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    pts = np.hstack([corners, ones])  # (4, 3)
    # 2x3 矩阵 * 3xN
    transformed = (M @ pts.T).T  # (4, 2)
    return corners_to_yolo(cls, transformed, img_w, img_h)


def transform_boxes(boxes, M, img_w, img_h):
    """
    对一组 YOLO boxes 应用仿射变换
    brightness/contrast/noise 用单位矩阵 M=identity 即可
    """
    new_boxes = []
    for box in boxes:
        new_box = apply_affine_to_box(box, M, img_w, img_h)
        if new_box is not None:
            new_boxes.append(new_box)
    return new_boxes


def main():
    img_paths = glob.glob(os.path.join(SRC_IMG_DIR, "*.jpg"))
    print(f"[信息] 找到 {len(img_paths)} 张测试集图片。")

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        name, ext = os.path.splitext(img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[警告] 无法读取图像：{img_path}")
            continue

        h, w = img.shape[:2]

        # 对应标签
        label_path = os.path.join(SRC_LABEL_DIR, f"{name}.txt")
        boxes = load_yolo_labels(label_path)

        # ========== 1. 亮度 / 对比度 / 噪声 (几何不变，M=单位矩阵) ==========
        I = np.float32([[1, 0, 0],
                        [0, 1, 0]])

        bright_img = adjust_brightness_contrast(img, alpha=1.0, beta=30)
        dark_img = adjust_brightness_contrast(img, alpha=1.0, beta=-30)
        contrast_img = adjust_brightness_contrast(img, alpha=1.3, beta=0)
        noise_img = add_gaussian_noise(img, std=15)

        bright_boxes = transform_boxes(boxes, I, w, h)
        dark_boxes = transform_boxes(boxes, I, w, h)
        contrast_boxes = transform_boxes(boxes, I, w, h)
        noise_boxes = transform_boxes(boxes, I, w, h)

        # 保存这四种
        cv2.imwrite(os.path.join(DST_IMG_DIR, f"{name}_bright{ext}"), bright_img)
        save_yolo_labels(os.path.join(DST_LABEL_DIR, f"{name}_bright.txt"), bright_boxes)

        cv2.imwrite(os.path.join(DST_IMG_DIR, f"{name}_dark{ext}"), dark_img)
        save_yolo_labels(os.path.join(DST_LABEL_DIR, f"{name}_dark.txt"), dark_boxes)

        cv2.imwrite(os.path.join(DST_IMG_DIR, f"{name}_contrast{ext}"), contrast_img)
        save_yolo_labels(os.path.join(DST_LABEL_DIR, f"{name}_contrast.txt"), contrast_boxes)

        cv2.imwrite(os.path.join(DST_IMG_DIR, f"{name}_noise{ext}"), noise_img)
        save_yolo_labels(os.path.join(DST_LABEL_DIR, f"{name}_noise.txt"), noise_boxes)

        # ========== 2. 旋转 ==========
        rot_img, M_rot = rotate_image_and_matrix(img, angle=8)
        rot_boxes = transform_boxes(boxes, M_rot, w, h)
        cv2.imwrite(os.path.join(DST_IMG_DIR, f"{name}_rot{ext}"), rot_img)
        save_yolo_labels(os.path.join(DST_LABEL_DIR, f"{name}_rot.txt"), rot_boxes)

        # ========== 3. 平移 ==========
        shift_img, M_shift = shift_image_and_matrix(img, dx=5, dy=5)
        shift_boxes = transform_boxes(boxes, M_shift, w, h)
        cv2.imwrite(os.path.join(DST_IMG_DIR, f"{name}_shift{ext}"), shift_img)
        save_yolo_labels(os.path.join(DST_LABEL_DIR, f"{name}_shift.txt"), shift_boxes)

    print(f"[完成] 增强图像和新标签已保存到：")
    print(f"  图像: {DST_IMG_DIR}")
    print(f"  标签: {DST_LABEL_DIR}")


if __name__ == "__main__":
    main()
