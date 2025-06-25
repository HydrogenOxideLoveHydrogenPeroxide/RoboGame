import os

from Camera import Camera
from util import recode
from Settings import Settings
from Identifier import Identifier
from connect import Connector
import cv2 as cv
import time,json

settings = Settings()
def main():
    camera = Camera()
    connector = Connector()
    identifier=Identifier()
    camera.open()

    # clear_files_in_directory(camera.dirpath)
    sep=settings.sep
    try:
        while True:
            # get a frame
            ret, frame = camera .cap.read()
            frame = cv.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
            frame = cv.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊)
            result,frame=identifier.main(frame)
            result=recode(result)
            print(json.dumps(result))
            cv.imshow('camera', frame)
            cv.waitKey(1)
            if connector.ser:
                connector.Port_send(json.dumps(result))#json对象字符串
            # 每隔sep秒截屏一次
            time.sleep(sep)
    except KeyboardInterrupt:
        print("捕捉到中断信号，结束程序。")

    camera.close()

if __name__ == '__main__':
    main()