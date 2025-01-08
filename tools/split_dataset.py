import os
import random
import shutil

# 设置路径
image_dir = '/home/chenhaiwei/resources/smart12.v2i.yolov8/train/images'  # 图片文件夹路径
label_dir = '/home/chenhaiwei/resources/smart12.v2i.yolov8/train/labels'  # 标注文件夹路径
train_image_dir = '/home/chenhaiwei/resources/smart/train/images'
train_label_dir = '/home/chenhaiwei/resources/smart/train/labels'
val_image_dir = '/home/chenhaiwei/resources/smart/val/images'
val_label_dir = '/home/chenhaiwei/resources/smart/val/labels'

# 获取所有图片文件
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]  # 根据实际图片后缀调整
total_images = len(image_files)

# 计算训练集和验证集的数量
train_count = int(0.8 * total_images)
val_count = total_images - train_count

# 打乱图片顺序
random.shuffle(image_files)

# 划分训练集和验证集
train_images = image_files[:train_count]
val_images = image_files[train_count:]

# 创建目录
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 将训练集和验证集图片及标注文件复制到相应目录
for image in train_images:
    shutil.copy(os.path.join(image_dir, image), os.path.join(train_image_dir, image))
    label_file = image.replace('.jpg', '.txt').replace('.png', '.txt')  # 根据实际图片后缀调整
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))

for image in val_images:
    shutil.copy(os.path.join(image_dir, image), os.path.join(val_image_dir, image))
    label_file = image.replace('.jpg', '.txt').replace('.png', '.txt')  # 根据实际图片后缀调整
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(val_label_dir, label_file))

print(f"数据集划分完成！训练集：{train_count} 张图片，验证集：{val_count} 张图片")
