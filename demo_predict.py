
from ultralytics import YOLO
from torchinfo import summary

yolo = YOLO(r"C:\Code\python\ultralytics-main\runs\C2f_Res2block+EMA+MHSA\weights\best.pt", task="detect")


# summary(yolo, input_size=(1, 3, 640, 640))

result = yolo(source=r"C:\Users\Haiwei_Chen\Documents\booty\Paper\Dataset\4.2k_HRW_yolo_dataset\images\trainjpg\0009001.jpg", save=True, conf=0.2, iou=0.8, show=True)
