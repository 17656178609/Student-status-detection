from ultralytics import YOLO

model = YOLO(r"D:\Code\pycharm\ultralytics-main\runs\detect\train4\weights\best.pt")

model.train(data='class.yaml', workers=0, epochs=100, batch=16)