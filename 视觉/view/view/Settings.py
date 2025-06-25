import cv2 as cv
class Settings:#设置
    def __init__(self):
        self.sep=0.1
        #巡航线识别的设置
        self.LineIdentifierSettings={
            'erode':{
                'iterations':8,
            },
            'dilate':{
                'iterations':2,
                'ksize':(5,5)
            }
        }

        self.CubeIdentifierSettings={
            'condition':{
                'lower':(0, 130, 50),
                'upper': (70, 255, 255)
            },
            'erode':{
                'iterations':5,
                'kenel':cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
            },
            'dilate':{
                'iterations':5,
                'ksize':(5,5)
            },
            'areaThreshold':500 #矩形面积阈值，小于这个面积会被忽略
        }

        self.QRcodeIdentifierSettings={
            'dilate': {
                'iterations': 2,
                'ksize': (3, 3)
            },
            "min_area":2000,
            "max_area":10000,
            "max_area_rate":1/3,#最大矩形面积占总面积
            "epsilon":1,#1-epsion<w/h<1+epsion
            "threshold":180,#二值化阈值
            'difference_thresh':120000,
            "screenCnt_epsilon":300,#位置误差估计阈值
            "default_size":(64,64),
            "source_type":{
                0:'QRcode/source/thresh/0.png',
                1:'QRcode/source/thresh/1.png',
                2:'QRcode/source/thresh/2.png',
                3:'QRcode/source/thresh/3.png',
                4:'QRcode/source/thresh/4.png',
                5:'QRcode/source/thresh/5.png',
            },
            'model_path':'model.pkl'
        }