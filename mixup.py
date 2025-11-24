import cv2
import numpy as np
import os
import random


def mixup_images(image1, image2, alpha, label1, label2):
    # 调整图像大小
    image1_resized = cv2.resize(image1, (int(image1.shape[1] * alpha), int(image1.shape[0] * alpha)))

    # 随机选择位置
    x = random.randint(0, image2.shape[1] - image1_resized.shape[1])
    y = random.randint(0, image2.shape[0] - image1_resized.shape[0])

    # 将缩小后的图像贴入目标图像
    mixed_image = image2.copy()
    mixed_image[y:y + image1_resized.shape[0], x:x + image1_resized.shape[1]] = cv2.addWeighted(
        image2[y:y + image1_resized.shape[0], x:x + image1_resized.shape[1]], 1.0, image1_resized, 0.5, 0)

    # 更新标签位置
    updated_label1 = label1.copy()
    updated_label2 = label2.copy()
    updated_label1[1] = int(label1[1] * alpha + x)
    updated_label1[2] = int(label1[2] * alpha + y)
    updated_label1[3] = int(label1[3] * alpha)
    updated_label1[4] = int(label1[4] * alpha)
    updated_label2[1] = int(label2[1] + x)
    updated_label2[2] = int(label2[2] + y)
    updated_label2[3] = int(label2[3])
    updated_label2[4] = int(label2[4])

    # 合并标签
    mixed_label = np.concatenate((updated_label1, updated_label2), axis=0)

    return mixed_image, mixed_label


# 设置参数
alpha = 0.5  # 缩小比例
input_dir = r'C:\Users\aa\Desktop\try\image\1'  # 输入图像文件夹路径
label_dir = r'C:\Users\aa\Desktop\try\labal\2'  # 输入标签文件夹路径
output_dir = r'C:\Users\aa\Desktop\output\image'  # 输出图像文件夹路径
output_label_dir = r'C:\Users\aa\Desktop\output\label'  # 输出标签文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)

# 遍历输入图像文件夹中的图像
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 读取图像
        image = cv2.imread(os.path.join(input_dir, filename))

        # 读取相应的标签
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)
        label = np.loadtxt(label_path, delimiter=' ')

        # 选择随机的另一张图像和标签
        random_filename = random.choice(os.listdir(input_dir))
        random_image = cv2.imread(os.path.join(input_dir, random_filename))
        random_label_filename = os.path.splitext(random_filename)[0] + '.txt'
        random_label_path = os.path.join(label_dir, random_label_filename)
        random_label = np.loadtxt(random_label_path, delimiter=' ')

        # 执行Mixup操作
        mixed_image, mixed_label = mixup_images(image, random_image, alpha, label, random_label)

        # 保存混合后的图像
        output_filename = 'mixed_' + filename
        cv2.imwrite(os.path.join(output_dir, output_filename), mixed_image)

        # 保存混合后的标签
        output_label_filename = 'mixed_' + label_filename
        with open(os.path.join(output_label_dir, output_label_filename), 'w') as f:
            f.write(f"{int(mixed_label[0])}")
            f.write(' ' + ' '.join(str(num) for num in mixed_label[1:5] / image.shape[1]))
            f.write('\n')
            f.write(f"{int(mixed_label[5])}")
            f.write(' ' + ' '.join(str(num) for num in mixed_label[6:10] / image.shape[1]))
            f.write('\n')
