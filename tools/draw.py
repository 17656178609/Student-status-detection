from PIL import Image, ImageDraw
import os
def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def parse_bounding_boxes(lines):
    """
    Parse bounding boxes from lines read from a .txt file.

    Each line is expected to contain:
    class_id x_center y_center width height (all normalized values)

    Args:
    - lines (list of str): List of lines read from a .txt file.

    Returns:
    - List of tuples: [(class_id, x_center, y_center, width, height), ...]
    """
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            # Parse the class_id and bounding box coordinates
            cla, x, y, w, h = map(float, parts)  # Convert all values to float
            boxes.append((cla, x, y, w, h))  # Add tuple to list
    return boxes


def process_folder_with_boxes(labels_path, images_path, outputs_path):
    for file_name in os.listdir(labels_path):
        if file_name.endswith('.txt'):
            print(f'处理文件: {file_name}')
            txt_file_path = os.path.join(labels_path, file_name)
            lines = read_txt_file(txt_file_path)
            bounding_boxes = parse_bounding_boxes(lines)
            image_name = os.path.splitext(file_name)[0] + '.jpg'
            image_path = os.path.join(images_path, image_name)

            print(f'对应的图片文件: {image_path}')
            if os.path.exists(image_path):
                print(f'找到对应的图片文件: {image_path}')
                image = Image.open(image_path)
                image_width, image_height = image.size

                # 创建绘制对象
                draw = ImageDraw.Draw(image)

                for i, box in enumerate(bounding_boxes):
                    cla, x_center, y_center, width, height = box

                    # Convert normalized coordinates to pixel values
                    x1 = int((x_center - width / 2) * image_width)
                    y1 = int((y_center - height / 2) * image_height)
                    x2 = int((x_center + width / 2) * image_width)
                    y2 = int((y_center + height / 2) * image_height)

                    # Draw the bounding box on the image
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                    # Optionally, you can also add class label text
                    label = str(int(cla))
                    draw.text((x1, y1 - 10), label, fill="red")

                # Save the image with bounding boxes drawn on it
                output_path = os.path.join(outputs_path, f'{os.path.splitext(file_name)[0]}_with_boxes.jpg')
                image.save(output_path)
                print(f'保存带框的图片: {output_path}')

# Example usage
labels_path = '/home/chenhaiwei/resources/VOCdevkit/labels/val'
images_path = '/home/chenhaiwei/resources/VOCdevkit/images/val'
outputs_path = '/home/chenhaiwei/resources/VOCdevkit/outputs/'
process_folder_with_boxes(labels_path, images_path, outputs_path)
