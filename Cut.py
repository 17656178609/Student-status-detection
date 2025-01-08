import os
from ultralytics import YOLO
from PIL import Image

# 1. 配置路径
images_folder = "/home/chenhaiwei/resources/4.2k_HRW_yolo_dataset/images/train"  # 输入图片文件夹路径
output_folder = "/home/chenhaiwei/resources/YOLOv8_Cut"  # 保存裁剪目标的文件夹路径
model_path = "/home/chenhaiwei/code/Student-status-detection/runs/detect/train/C2f_Res2block+EMA+MHSA/weights/best.pt"  # 预训练模型路径，可以替换为你的自定义模型

# 2. 加载 YOLOv8 模型
model = YOLO(model_path)

# 3. 创建输出目录（若不存在）
os.makedirs(output_folder, exist_ok=True)

# 4. 遍历图片并裁剪目标
for img_name in os.listdir(images_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(images_folder, img_name)
        print(f"Processing {img_path}...")
        
        # 运行 YOLOv8 推理
        results = model(img_path)  # 推理
        
        # 5. 提取检测到的目标并裁剪
        for i, result in enumerate(results):
            boxes = result.boxes  # 检测框
            img = Image.open(img_path)
            
            for j, box in enumerate(boxes):
                # 获取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                cropped_img = img.crop((x1, y1, x2, y2))  # 裁剪图像
                
                # 保存裁剪后的目标
                crop_name = f"{os.path.splitext(img_name)[0]}_crop_{i}_{j}.png"
                output_path = os.path.join(output_folder, crop_name)
                cropped_img.save(output_path)
                print(f"Saved cropped image: {output_path}")

print("All images processed successfully!")
