import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 你的原图和mask路径
img_dir = 'cvclindb/PNG/Original'
mask_dir = 'cvclindb/PNG/Ground Truth'

# YOLO需要的保存目录
save_root = 'datasets/cvc/'
save_img_dir = os.path.join(save_root, 'images')
save_label_dir = os.path.join(save_root, 'labels')

# 创建目录
for split in ['train', 'val']:
    os.makedirs(os.path.join(save_img_dir, split), exist_ok=True)
    os.makedirs(os.path.join(save_label_dir, split), exist_ok=True)

# 获取所有图片名
all_images = sorted(os.listdir(img_dir))
train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

def mask_to_yolo(mask_path, image_shape):
    mask = cv2.imread(mask_path, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:  # 防止小噪声
            continue
        cx = (x + w/2) / image_shape[1]
        cy = (y + h/2) / image_shape[0]
        bw = w / image_shape[1]
        bh = h / image_shape[0]
        labels.append([0, cx, cy, bw, bh])
    return labels

def process(images, split):
    for img_name in tqdm(images):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        labels = mask_to_yolo(mask_path, (h, w))

        # 保存图片
        cv2.imwrite(os.path.join(save_img_dir, split, img_name), img)

        # 保存标签
        txt_name = img_name.replace('.png', '.txt')
        with open(os.path.join(save_label_dir, split, txt_name), 'w') as f:
            for label in labels:
                f.write(' '.join([str(round(x, 6)) for x in label]) + '\n')

# 执行划分和转换
process(train_images, 'train')
process(val_images, 'val')