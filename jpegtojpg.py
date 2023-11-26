import os
from PIL import Image

train_folder = "D:\Dataset\hulianwang+\images"


for img_name in os.listdir(train_folder):
    if img_name[-4:] == 'jpeg':
        cur_img = os.path.join(train_folder, img_name)
        img_new_name = img_name.split(".")[0] + '.jpg'
        img_new_path = os.path.join(train_folder, img_new_name)
        src_img = Image.open(cur_img)
        src_img.save(img_new_path, quality=95)
        os.remove(cur_img) #会删除原始文件
        print("{} saved".format(img_new_name))