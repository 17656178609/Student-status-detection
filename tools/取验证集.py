import os
from PIL import Image
from shutil import copy2
ori_folder = r"D:\Dataset\学习行为\da"
new_folder = r"D:\Dataset\学习行为"
if os.path.isdir(new_folder):
    pass
else:
    os.mkdir(new_folder)
cont = 0
for img_name in os.listdir(ori_folder):
    if cont % 5 == 0:
        cur_img = os.path.join(ori_folder, img_name)
        img_new_path = os.path.join(new_folder, img_name)
        print(cur_img + " " + img_new_path)
        copy2(cur_img, img_new_path)
        os.remove(cur_img) #会删除原始文件
        print("{} saved".format(img_name))
    cont += 1