import os
from PIL import Image

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_bounding_boxes(lines):
    bounding_boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            bounding_boxes.append(tuple(map(float, parts)))
    return bounding_boxes


def process_folder(labels_path, images_path, outputs_path):
    for file_name in os.listdir(labels_path):
        if file_name.endswith('.txt'):
            print(f'处理文件: {file_name}')
            txt_file_path = os.path.join(labels_path, file_name)
            lines = read_txt_file(txt_file_path)
            bounding_boxes = parse_bounding_boxes(lines)
            image_name = os.path.splitext(file_name)[0] + '.jpg'
            image_path = os.path.join(images_path, image_name)

            if os.path.exists(image_path):
                print(f'找到对应的图片文件: {image_path}')
                image = Image.open(image_path)
                image_width, image_height = image.size

                for i, box in enumerate(bounding_boxes):
                    cla, x_center, y_center, width, height = box
                    expand_ratio = 0.2
                    expand_width = width * expand_ratio
                    expand_height = height * expand_ratio
                    # Convert normalized coordinates to pixel values
                    # x1 = int((x_center - width / 2) * image_width)
                    # y1 = int((y_center - height / 2) * image_height)
                    # x2 = int((x_center + width / 2) * image_width)
                    # y2 = int((y_center + height / 2) * image_height)
                    x1 = int((x_center - width / 2) * image_width - expand_width / 2)
                    y1 = int((y_center - height / 2) * image_height - expand_height / 2)
                    x2 = int((x_center + width / 2) * image_width + expand_width / 2)
                    y2 = int((y_center + height / 2) * image_height + expand_height / 2)
                    x1 = max(0, min(x1, image_width - 1))
                    y1 = max(0, min(y1, image_height - 1))
                    x2 = max(0, min(x2, image_width - 1))
                    y2 = max(0, min(y2, image_height - 1))
                    # Create output folder based on class
                    output_folder = os.path.join(outputs_path, str(int(cla)))
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Output file name and path
                    output_path = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}_{i}.jpg')

                    if x1 < x2 and y1 < y2:  # 确保裁剪区域有效
                        cropped_image = image.crop((x1, y1, x2, y2))
                        cropped_image.save(output_path)
                        print(f"保存裁剪后的图像: {output_path}")
                    else:
                        print(f"无效的边界框: {x1}, {y1}, {x2}, {y2}")


# Example usage
labels_path = '/home/chenhaiwei/resources/smart/val/labels'
images_path = '/home/chenhaiwei/resources/smart/val/images'
outputs_path = '/home/chenhaiwei/resources/smart/outputs/val'
process_folder(labels_path, images_path, outputs_path)
