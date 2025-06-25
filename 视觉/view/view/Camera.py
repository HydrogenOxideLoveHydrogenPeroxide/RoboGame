# author:yrq
# date:2024/8/8 20:05
# comment:这是调用计算机的摄像机
import logging,os,time
import cv2 as cv
from pathlib import Path

from util import clear_files_in_directory

class Camera(object):
    def __init__(self):
        self.dirpath=Path('temp')#临时存储得到的照片
        self.limit=100

    def open(self):#打开摄像头
        self.cap = cv.VideoCapture(0,cv.CAP_DSHOW)  # 0为电脑内置摄像头
        if not self.cap.isOpened():#如果打开失败
            logging.info("无法打开摄像头")
            exit(200)
    def close(self):
        self.cap.release()
        cv.destroyAllWindows()
    def photoshot(self,index,frame):#截屏操作

        filename = os.path.join('temp', 'photo' + str(index) + '.png')# 生成文件名
        logging.info("图片名称{}".format(filename))
        cv.imwrite(filename, frame)  # 保存路径

    def photoshotCeaselessly(self,sep=0.5):#连续截屏,代码之间的间隔
        """
            sep:每隔多长时间截屏一次，default:0.1s
        :param sep:
        :return:
        """
        clear_files_in_directory(self.dirpath)
        index=0
        try:
            while True:
                # get a frame
                ret, frame = self.cap.read()
                frame = cv.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
                # frame = cv.resize(frame, (0, 0), fx=0.3, fy=0.3)
                frame = cv.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊)

                self.photoshot(index,frame)
                #index更新
                index+=1
                index%=self.limit

                # 每隔100毫秒截屏一次
                time.sleep(sep)
        except KeyboardInterrupt:
            print("捕捉到中断信号，结束程序。")

    def test(self):#测试代码
        while (True):
            ret, frame = self.cap.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
            frame = cv.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
            cv.imshow("video", frame)
            esc = cv.waitKey(50)#按下esc键
            if esc == 27:
                break
        cv.destroyAllWindows()

if __name__=="__main__":
    camera = Camera()
    camera.open()
    camera.photoshotCeaselessly()
    camera.close()