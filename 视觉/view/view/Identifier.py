# author:yrq
# date:2024/8/8 20:05
# comment:这是识别图片的方块和巡航线部分
import logging
import os.path
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from pyautogui import *
import imutils
from scipy.spatial.distance import euclidean

from util import  clear_files_in_directory
from Settings import Settings
import joblib

#基本配置
srcDir=Path('temp')
resultDir=Path('result')
debug=False
settings=Settings()
ScalePercent=50

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Identifier:#统一管理identify
    def __init__(self):
        pass

    def main(self,frame):
        orig=frame.copy()#在这个图上识别，frame上修改

        cubeIdentifier = CubeIdentifier()
        qr_indentifier = QRcodeIdentifier()

        cubes=cubeIdentifier.main(orig,frame)
        qrcodes=qr_indentifier.main(orig,frame)

        return {
            'cube':cubes[0],
            'qrcode':qrcodes[0],
        },frame

    def show(self,image, filename=None, if_show=True):  # 如果有文件名，默认输出检测到的结果为新的图片
        # 调整窗口大小
        scale_factor = 1
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dim = (width, height)
        resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

        if if_show:
            # 显示图片
            cv.imshow('QR Code Square', resized_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        def new_filename_genertater(file_path):
            # 获取文件的目录路径
            directory = os.path.dirname(file_path)
            # 获取文件名（包括扩展名）
            filename = os.path.basename(file_path)
            filename_mid = os.path.splitext(filename)

            os.path.exists("QRcode/processed") or os.makedirs("QRcode/processed")

            newfilename = Path(os.path.dirname(directory)) / "processed" / (
                        f"{filename_mid[0]}_processed" + filename_mid[1])
            print(newfilename)
            return str(newfilename)

        # 保存图片
        if filename:
            newfilename = new_filename_genertater(filename)

            cv.imwrite(newfilename, image)

    def merge_images(self,image1, image2):
        # 确保两张图片尺寸相同
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions")

        # 创建一个空白图片，用于存储合并结果
        merged = np.copy(image1)

        # 遍历每个像素
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                # 检查两个像素是否相同
                if np.array_equal(image1[i, j], image2[i, j]):
                    continue  # 相同则保留image1的像素
                else:
                    # 不相同，检查image1的像素是否为绿色
                    if image1[i, j][1] > image1[i, j][0] and image1[i, j][1] > image1[i, j][2]:
                        continue  # 是绿色则保留image1的像素
                    else:
                        merged[i, j] = image2[i, j]  # 否则用image2的像素覆盖

        return merged

class CubeIdentifier():
    def __init__(self):
        self.condition=settings.CubeIdentifierSettings['condition']
        self.erode_settings=settings.CubeIdentifierSettings['erode']
        self.dilate_settings=settings.CubeIdentifierSettings['dilate']
        self.areaThreshold=settings.CubeIdentifierSettings['areaThreshold']

    def convert(self,frame):#处理加工图像
        # input=srcDir/filename

        # img = cv.imread(input)  # 读取彩色图像

        try:
            img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # 转化为 HSV 格式
            original = img
            img = cv.GaussianBlur(img, (5, 5), 0)
            img = cv.erode(img, self.erode_settings['kenel'], iterations=self.erode_settings['iterations'])  # 腐蚀 粗的变细
            img = cv.inRange(img, self.condition['lower'], self.condition['upper'])#mask
            mask=img
            if self.dilate_settings.get('iterations',0):
                kernel = cv.getStructuringElement(cv.MORPH_RECT, self.dilate_settings.get('ksize')) # 创建一个3x3的矩形结构元素
                img = cv.dilate(img, kernel, iterations=self.dilate_settings.get('iterations',0))# 膨胀操作

            img_output= img
        except cv.error:
            logging.error(f"CubeIdentifier.convert()处理文件{input}发生错误",exc_info=True)
            return

        #保存调试用
        # self.bitwise(original,img_output)
        return img

    def test(self,image,filename=None,show=True,tofig=True,mode='general'):
        if mode=='general':
            scale_percent = ScalePercent  # 缩小比例
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
            if show:
                cv.imshow('Image Window', image)
                cv.waitKey(0)
                cv.destroyAllWindows()
            not tofig or cv.imwrite(filename, image)
        else:
            plt.imshow(image, cmap='gray')
            if tofig:
                plt.savefig(filename)
            if show:
                plt.axis('off')  # 去除坐标轴
                plt.show()

    def rectContours(self, image, frame):
        cnts = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        contours_info = []

        # 图像底边中点
        image_center = (frame.shape[1] / 2, frame.shape[0])

        for c in cnts:
            if cv.contourArea(c) > self.areaThreshold:  # 忽略小区域
                rect = cv.minAreaRect(c)  # 最小外接矩形
                box = cv.boxPoints(rect)
                box = np.int64(box)

                # 绘制矩形框
                cv.drawContours(frame, [box], 0, (0, 255, 0), 2)

                # 计算方位角和距离
                center = rect[0]
                angle_rad = np.arctan2(image_center[1] - center[1], image_center[0] - center[0])
                angle = np.degrees(angle_rad)
                distance = self.calculate_distance(center, frame.shape)

                # 绘制从底边中点到方块中心的连线
                cv.line(frame, (int(image_center[0]), int(image_center[1])),
                        (int(center[0]), int(center[1])), (0, 0, 255), 2)  # 红色线条

                # 绘制方位角
                angle_text = f"{angle:.2f}"
                text_position = (int(center[0]) + 10, int(center[1]) - 10)  # 文本位置略偏移
                cv.putText(frame, angle_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                           cv.LINE_AA)  # 红色文本

                contours_info.append((int(angle), int(distance)))

        return frame, contours_info

    def calculate_distance(self, center, image_shape):
        # 将图像的中心点设为底边中间，即 (image_shape[1] / 2, image_shape[0])
        image_center = (image_shape[1] / 2, image_shape[0])

        # 计算物体中心点与底边中间点的距离
        distance = np.sqrt((center[0] - image_center[0]) ** 2 + (center[1] - image_center[1]) ** 2)

        return distance

    def bitwise(self,image,mask,filename):#抠图
        result=cv.bitwise_and(image,image,mask=mask)

        scale_percent = ScalePercent # 缩小比例
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        result = cv.resize(result, dim, interpolation=cv.INTER_AREA)

        output = resultDir / (os.path.splitext(filename)[0] + '_output' + '.png')
        cv.imwrite(output, result)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def display_combined(self, mask, result_image):
        # 转换掩码为三通道图像以便于拼接
        mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        # 确保两个图像大小一致
        if mask_rgb.shape != result_image.shape:
            mask_rgb = cv.resize(mask_rgb, (result_image.shape[1], result_image.shape[0]))

        # 横向拼接图像
        combined_image = np.hstack((mask_rgb, result_image))
        scale_percent = ScalePercent  # 缩小比例
        width = int(combined_image.shape[1] * scale_percent / 100)
        height = int(combined_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        combined_image = cv.resize(combined_image, dim, interpolation=cv.INTER_AREA)

        # 显示拼接图像
        # cv.imshow('Mask and Result', combined_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return combined_image

    def main(self,orig,frame):
        # 处理图像
        original_image=orig#原图，用来识别
        img_output = self.convert(frame)#用来画图

        if img_output is None:
            return

        # 获取轮廓和信息
        result_image, contours_info = self.rectContours(img_output, frame)

        # 显示结果
        combined=self.display_combined(img_output, result_image)

        # 输出方位角和距离信息
        for idx, (angle, distance) in enumerate(contours_info):
            logging.debug(f"方块 {idx + 1}:, 方位角={angle}, 距离={distance}")

        return contours_info,frame

class LineIdentifier():
    def __init__(self):
        self.erode_settings = settings.LineIdentifierSettings.get("erode")
        self.dilate_settings =settings.LineIdentifierSettings.get("dilate")

    def convert(self,frame):
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
        binary = cv.erode(binary, None, iterations=self.erode_settings.get('iterations',0))  # 腐蚀 粗的变细
        if self.dilate_settings.get('iterations', 0):
            kernel = cv.getStructuringElement(cv.MORPH_RECT, self.dilate_settings.get('ksize'))  # 创建一个3x3的矩形结构元素
            binary = cv.dilate(binary, kernel, iterations=self.dilate_settings.get('iterations', 0))  # 膨胀操作
        return binary,frame

    def judgePosition(self,binary,image):#判断黑线是对的还是错的
        # 找到轮廓
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # 假设黑线是最大轮廓
        largest_contour = min(contours, key=cv.contourArea)

        # 计算轮廓的边界框
        x, y, w, h = cv.boundingRect(largest_contour)

        # 计算黑线的中心点
        center_x = x + w // 2

        # 获取图像宽度
        image_width = image.shape[1]

        # 计算中间1/3部分的左右边界
        left_boundary = image_width // 3
        right_boundary = 2 * image_width // 3

        # 判断黑线的中心点位置
        if center_x < left_boundary:
            position = 'L'  # 左侧
        elif center_x > right_boundary:
            position = 'R'  # 右侧
        else:
            position = 'C'  # 中间

        # 画出轮廓
        cv.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)

        # 在图像上标注出黑线的位置
        cv.putText(image, f'Position: {position}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示图像
        scale_percent = 25  # 缩小比例
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        if debug:
            cv.imshow('Image with Contour', image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        positionToNum=lambda x: 1 if x == 'R' else -1 if x == 'L' else 0
        return positionToNum(position)

    def test(self,image,filename=None,show=True,tofig=True,mode='general'):
        if mode=='general':
            scale_percent = 25  # 缩小比例
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
            if show:
                cv.imshow('Image Window', image)
                cv.waitKey(0)
                cv.destroyAllWindows()
            not tofig or cv.imwrite(filename, image)
        else:
            plt.imshow(image, cmap='gray')
            if tofig:
                plt.savefig(filename)
            if show:
                plt.axis('off')  # 去除坐标轴
                plt.show()

    def main(self,frame):#这个类的主程序，运行这个
        binary,image=self.convert(frame)
        return self.judgePosition(binary,image)

class QRcodeIdentifier():
    def __init__(self):#放置相关参数
        self.QRcodeSettings=settings.QRcodeIdentifierSettings
        self.dilate_settings = self.QRcodeSettings.get('dilate')
        model_path=self.QRcodeSettings.get("model_path")
        self.model=joblib.load(model_path)
        self.PCA_model=joblib.load('PCA.joblib') # 加载pca模型
    def convert(self, frame):
        # 加载原图 并对原图进行Resize
        image = frame
        image = cv.GaussianBlur(image, (5, 5), 0)#高斯滤波
        height=image.shape[0]
        self.ratio = image.shape[0] / height
        self.orig = image.copy()
        image = imutils.resize(image, height=height)  # 根据长宽比自动计算另外一边的尺寸进行resize
        # 根据处理找到边缘
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        edged = cv.Canny(gray, 75, 200)
        return image,gray,edged

    def show(self,image, filename=None, if_show=True):  # 如果有文件名，默认输出检测到的结果为新的图片
        # 调整窗口大小
        scale_factor = 1
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dim = (width, height)
        resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

        if if_show:
            # 显示图片
            cv.imshow('QR Code Square', resized_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        def new_filename_genertater(file_path):
            # 获取文件的目录路径
            directory = os.path.dirname(file_path)
            # 获取文件名（包括扩展名）
            filename = os.path.basename(file_path)
            filename_mid = os.path.splitext(filename)

            os.path.exists("QRcode/processed") or os.makedirs("QRcode/processed")

            newfilename = Path(os.path.dirname(directory)) / "processed" / (
                        f"{filename_mid[0]}_processed" + filename_mid[1])
            print(newfilename)
            return str(newfilename)

        # 保存图片
        if filename:
            newfilename = new_filename_genertater(filename)

            cv.imwrite(newfilename, image)

    def exist(self,screenCnt,screenCnts):
        for contract_screenCnt in screenCnts:
            # 将两个轮廓的顶点转换为 numpy 数组
            points1 = np.array(screenCnt, dtype=np.float32)
            points2 = np.array(contract_screenCnt, dtype=np.float32)
            # 计算每个对应顶点对之间的距离
            distances = np.linalg.norm(points1 - points2, axis=1)

            # 检查所有顶点对的距离是否都小于阈值
            if np.all(distances < self.QRcodeSettings.get("screenCnt_epsilon")):
                return True
        return False

    def find_qr_code(self,image,edged):
        screenCnts=[]
        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)  ## 找到轮廓
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]#提取面积前10

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)
            area = cv.contourArea(approx)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4 and area>self.QRcodeSettings.get("min_area"):
                screenCnt = approx
                if self.exist(screenCnt,screenCnts):#判断是不是已经在里面
                    continue
                screenCnts.append(screenCnt.copy())
                cv.drawContours(image, [screenCnt], -1, (0,0, 255), 2)
                
        return screenCnts,image#返回矩阵四个顶点位置的列表

    ####图像矫正部分###########
    def order_points(self,pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def four_point_transform(self,image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 选择正方形的边长
        squareSize = max(maxWidth, maxHeight)
        # 计算正方形的中心点
        center = np.array([(tl[0] + tr[0]) // 2, (tl[1] + tr[1]) // 2])
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        # dst = np.array([
        #     [0, 0],
        #     [maxWidth - 1, 0],
        #     [maxWidth - 1, maxHeight - 1],
        #     [0, maxHeight - 1]], dtype="float32")
        # 定义目标点
        dst = np.array([
            [0, 0],
            [squareSize - 1, 0],
            [squareSize - 1, squareSize - 1],
            [0, squareSize - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv.getPerspectiveTransform(rect, dst)
        # warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

        # 计算透视变换矩阵
        M = cv.getPerspectiveTransform(rect, dst)

        # 应用透视变换
        warped = cv.warpPerspective(image, M, (squareSize, squareSize))
        # return the warped image
        return warped


    #########################

    ###比较差异，判断种类
    def calculate_difference(self,image):
        def resize_image(image, size=(64,64)):
            # 调整图片尺寸
            resized_image = cv.resize(image, size)
            return resized_image

        default_size = self.QRcodeSettings.get("default_size")
        image = cv.resize(image, default_size)  # 假设我们将所有图像缩放到
        image_array=image.flatten()#把图片平整到一列

        data_pca = self.PCA_model.transform(np.array([image_array]))
        predictions = self.model.predict(data_pca)
        return predictions[0]
        # default_size=self.QRcodeSettings.get("default_size")
        # thresh=self.QRcodeSettings.get("difference_thresh")
        # # 将图片转换为浮点数以避免数据类型溢出
        # differences=[]
        #
        # for i,srcfile in self.QRcodeSettings.get("source_type").items():
        #     goal_image = resize_image(image,size=default_size).astype(np.float64)
        #     contract_image=cv.imread(srcfile)
        #     contract_image = resize_image(contract_image, size=default_size).astype(np.float64)
        #
        #     # 计算两张图片的差值
        #     difference = np.linalg.norm(goal_image - contract_image)
        #     differences.append(difference)
        # logging.info(differences)
        # if min(differences)<thresh:
        #     return differences.index(min(differences))
        # else:
        #     print(min(differences))
        #     return None

    def extract_data(self,screenCnts):
        class Rect:
            def __init__(self,screenCnt,type):
                self.rect_points=screenCnt
                self.type=type

            def find_polygon_center(self,points):
                """
                找到由四个顶点定义的矩形的中心点。
                参数:
                points: 一个包含四个顶点坐标的数组，形状为(4, 2)。
                返回:
                center: 矩形的中心点坐标。
                """
                # 将点的列表转换为numpy数组
                points = np.array(points)

                # 计算中心点
                center_of_mass = np.mean(points, axis=0)
                return center_of_mass

            def __call__(self, *args, **kwargs):
                return (self.find_polygon_center(self.rect_points),self.type)

        def export():#导出此时图片
            i=1
            maindir=Path("QRcode")/'train'
            currentfiles=os.listdir(maindir)
            while f'{i}.jpg' in currentfiles:
                i+=1
            cv.imwrite(str(maindir/f'{i}.jpg'), binary)


        Rects=[]
        type_existing=[]
        for screenCnt in screenCnts:
            warped = self.four_point_transform(self.orig, screenCnt.reshape(4, 2) * self.ratio)
            _, binary = cv.threshold(warped, self.QRcodeSettings.get("threshold"), 255, cv.THRESH_BINARY_INV)
            if self.dilate_settings.get('iterations', 0):
                kernel = cv.getStructuringElement(cv.MORPH_RECT, self.dilate_settings.get('ksize'))  # 创建一个3x3的矩形结构元素
                binary  = cv.dilate(binary , kernel, iterations=self.dilate_settings.get('iterations', 0))  # 膨胀操作

            binary = cv.bitwise_not(binary)# 反色

            type=self.calculate_difference(binary)

            if (type in type_existing and type):
                continue
            # self.show(binary)

            rect=Rect(screenCnt,type)
            type_existing.append(type)
            Rects.append(rect)
        return Rects

    def viewable(self,points,image):
        height, width, _ = image.shape# 获取图像的尺寸

        bottom_middle_x = width // 2# 计算底部中间点的坐标
        bottom_middle_y = height
        points = np.array(points,dtype=np.int32)

        for point in points:
            # 第一个参数是图像，第二个参数是点的坐标，第三个参数是半径，第四个参数是颜色，第五个参数是线条的宽度
            cv.circle(image, (point[0],point[1]), radius=5, color=(0, 0, 255), thickness=-1)

            # 绘制线
            # 第一个参数是图像，第二个参数是起点坐标，第三个参数是终点坐标，第四个参数是颜色，第五个参数是线条的宽度
            cv.line(image, (point[0],point[1]), (bottom_middle_x, bottom_middle_y), color=(0, 0, 255), thickness=2)

    def angle_distance(self,center,image_center): #视角出发点
        def calculate_distance(center):
            distance = np.sqrt((center[0] - image_center[0]) ** 2 + (center[1] - image_center[1]) ** 2)# 计算物体中心点与底边中间点的距离
            return distance

        angle_rad = np.arctan2(image_center[1] - center[1], image_center[0] - center[0])
        angle = np.degrees(angle_rad)
        distance = calculate_distance(center)

        return (int(angle), int(distance))

    def main(self,orig,frame):
        # self.show(frame)
        orig,gray,edged=self.convert(orig)#原图和灰度图
        screenCnts,frame=self.find_qr_code(frame,edged)
        qrcodes=self.extract_data(screenCnts)

        image_center = (frame.shape[1] / 2, frame.shape[0])
        QRcodes=[(self.angle_distance(qrcode()[0][0],image_center),int(qrcode()[1])) for qrcode in qrcodes]

        points=[qrcode()[0][0] for qrcode in qrcodes]
        # print(QRcodes)
        self.viewable(points,frame)
        # self.show(frame)
        return QRcodes,frame
    
if __name__=='__main__':
    filename='IMG_20240921_173609.jpg'
    img=cv.imread(filename)
    qr=QRcodeIdentifier()
    frame=img.copy()
    print(qr.main(img,frame)[0])
