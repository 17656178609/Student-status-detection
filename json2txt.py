import json
import os

def convert_json_to_yolov8(json_file_path, output_folder):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        image_filename = entry['ID']
        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        txt_filepath = os.path.join(output_folder, txt_filename)

        with open(txt_filepath, 'w') as txt_file:
            gtboxes = entry['gtboxes']
            for gtbox in gtboxes:
                tag = gtbox['tag']
                vbox = gtbox['vbox']
                x, y, w, h = vbox

                # Convert vbox coordinates to YOLO format
                x_center = x + w / 2
                y_center = y + h / 2
                x_center /= image_width
                y_center /= image_height
                w /= image_width
                h /= image_height

                # Write YOLO format annotation to .txt file
                txt_file.write(f"{tag} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    # Update these paths accordingly
    json_file_path = r"C:\Code\python\ultralytics-main\datasets\crowdhuman\images\train\train.json"
    output_folder = r"C:\Code\python\ultralytics-main\datasets\crowdhuman\images\val_a\train"
    
    convert_json_to_yolov8(json_file_path, output_folder)
