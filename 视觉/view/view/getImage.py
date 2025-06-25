import cv2
import os
import time

# 设置保存图像的目录
save_dir = 'QRcode/5'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 是默认摄像头

# 确保摄像头打开
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# 计数器，用于生成图像文件名
counter = 0

try:
    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        # 如果图像读取成功
        if ret:
            # 保存图像到指定目录
            img_name = os.path.join(save_dir, f'image_{counter}.jpg')
            cv2.imwrite(img_name, frame)
            print(f"Image {counter} saved as {img_name}")

            # 更新计数器
            counter += 1

            # 等待 3 秒
            time.sleep(3)
        else:
            print("Error: Could not read image.")
            break
except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()
