import cv2, os

def extract_frames(video_path, output_folder, num):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    print(success)
    count = num
    while success:
        cv2.imwrite(f"{output_folder}/image-{str(count).zfill(4)}.jpg", image)  # Save frame as JPEG file
        success, image = vidcap.read()
        count += 1
def main():

    # 调用函数，传入视频文件路径和输出文件夹路径
    lujing = r"C:\Users\Haiwei_Chen\Desktop\1"
    output_folder = r"C:\Users\Haiwei_Chen\Desktop\10"
    # file = os.listdir(lujing)
    num = 0
    # for file_cur in file:
    #     file_cur = os.path.join(lujing, file_cur)
    video = cv2.VideoCapture(r"C:\Users\Haiwei_Chen\Desktop\0\HandWash_001_A_11_G_01.mp4")
    save_step = 30
    while True:
        ret, frame = video.read()
        if not ret:
            break
        num += 1
        if num % save_step == 0:
            cv2.imwrite(f"{output_folder}/2-{str(num/save_step).zfill(4)}.jpg", frame)
            
if __name__ == '__main__':
    main()