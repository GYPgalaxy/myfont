import os
import cv2
import sys
import numpy as np
from .point import Point
from enum import Enum
from matplotlib import pyplot as plt
import datetime
from PIL import Image
sys.setrecursionlimit(1000000)

def read_images(path):
    img_names = []
    img_prefix = ['bmp', 'png', 'jpg', 'jpeg']
    files = os.listdir(path)
    for file in files:
        index = file.find('.')
        prefix = file[index+1:]
        if prefix in img_prefix:
            img_names.append(file)
    return img_names


def catImg(img_path_list, img_row, img_col, img_save_path, img_size=360):
    img_list = [Image.open(f) for f in img_path_list]
    _img_list = []
    for img in img_list:
        _img = img.resize((img_size, img_size), Image.BILINEAR)
        _img_list.append(_img)
    result = Image.new(_img_list[0].mode, (img_col*img_size, img_row*img_size))
    for i in range(img_row):
        for j in range(img_col):
            result.paste(_img_list[i*img_col+j], (j*img_size, i*img_size))
    result.save(img_save_path)

class MyImgUtil():
    def __init__(self, img_path):
        self.WHITE = 255
        self.BLACK = 0
        self.MAX_LENGTH = 15
        self.MIN_LENGTH = 3
        self.image_initial = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        _, th2 = cv2.threshold(self.image_initial, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.image_initial = th2
        self.row_num = self.image_initial.shape[0]  # 图像的行数
        self.col_num = self.image_initial.shape[1]  # 图像的列数
        self.Flag = [[False for i in range(self.col_num)] for i in range(self.row_num)]  # 标记数组
        self.Curves = []  # 闭合轮廓曲线钟点的坐标数组
        self.Trend = []  # 相邻点之间的趋势数组
        self.TurningPoints = []  # 拐点的坐标数组
        self.ControlPoints = []  # 控制点的坐标数组


    # 存储点(row,col)所在的闭合轮廓中每个点的坐标至Curve[index]中
    def saveOneCurve(self, index, row, col, HEIGHT, WIDTH):
        # 将点（row, col）存入Curve数组
        self.Curves[index].append(Point(row, col))
        self.Flag[row][col] = True
        delta_r = [-1, -1, 0, 1, 1, 1, 0, -1]
        delta_c = [0, 1, 1, 1, 0, -1, -1, -1]
        for i in range(8):
            r = row + delta_r[i]
            c = col + delta_c[i]
            if r > -1 and r < HEIGHT and c > -1 and c < WIDTH and not (r == row and c == col):
                if self.image_initial.item(r, c) == self.BLACK:
                    if not self.Flag[r][c]:
                        self.Trend[index].append(i + 1)
                        self.saveOneCurve(index, r, c, HEIGHT, WIDTH)


    # 存储图像img中的所有闭合轮廓曲线中所有点的坐标
    def saveAllCurves(self, img):
        index = 0  # 闭合轮廓的序号
        for row in range(self.row_num):
            for col in range(self.col_num):
                if self.image_initial.item(row, col) == self.BLACK:
                    if not (self.Flag[row][col]):
                        self.Curves.append([])
                        self.Trend.append([])
                        self.saveOneCurve(index, row, col, self.row_num, self.col_num)
                        index = index + 1


    # 寻找控制点的方法之一
    def findTurningPoints2(self):
        curvature = 0  # 曲率
        for x in range(len(self.Curves)):
            self.TurningPoints.append([])
            self.TurningPoints[x].append(0)
            y = 0
            while y < len(self.Curves[x]):
                # 直到这个曲率满足一定的值，停止，更换起始点，步长length清零
                length = self.MIN_LENGTH
                while length < self.MAX_LENGTH and length < len(self.Curves[x]) - y:
                    curvature = length / self.Curves[x][y].distance(self.Curves[x][y + length])
                    if curvature > 1.25:
                        self.TurningPoints[x].append(y + length)
                        y = y + length - 1
                        break
                    if length == self.MAX_LENGTH - 1:
                        self.TurningPoints[x].append(y + length)
                        y = y + length
                    length = length + 1
                y = y + 1


    # 通过拐点以及三阶贝塞尔曲线公式，计算出剩下的两个控制点
    def findControlPoints(self):
        Q0_index = Q1_index = Q2_index = Q3_index = numBetween = 0
        P1 = Point()
        P2 = Point()
        for i in range(len(self.TurningPoints)):
            self.ControlPoints.append([])
            for j in range(len(self.TurningPoints[i])):
                Q0_index = self.TurningPoints[i][j]
                if j == len(self.TurningPoints[i]) - 1:
                    Q3_index = self.TurningPoints[i][0]
                    numBetween = j - Q0_index
                else:
                    Q3_index = self.TurningPoints[i][j + 1]
                    numBetween = abs(Q3_index - Q0_index)
                if numBetween > 2:
                    Q1_index = Q0_index + numBetween // 3
                    Q2_index = Q0_index + 2 * numBetween // 3
                try:
                    Q0 = self.Curves[i][Q0_index]
                    Q1 = self.Curves[i][Q1_index]
                    Q2 = self.Curves[i][Q2_index]
                    Q3 = self.Curves[i][Q3_index]
                except IndexError:
                    return False
                P1.x = -5.0 / 6.0 * Q0.x + 3.0 * Q1.x - 1.5 * Q2.x + Q3.x / 3.0
                P1.y = -5.0 / 6.0 * Q0.y + 3.0 * Q1.y - 1.5 * Q2.y + Q3.y / 3.0
                P2.x = 1.0 / 3.0 * Q0.x - 1.5 * Q1.x + 3.0 * Q2.x - 5.0 * Q3.x / 6.0
                P2.y = 1.0 / 3.0 * Q0.y - 1.5 * Q1.y + 3.0 * Q2.y - 5.0 * Q3.y / 6.0
                self.ControlPoints[i].append(Point(P1.x, P1.y))
                self.ControlPoints[i].append(Point(P2.x, P2.y))


    def getResult(self):
        self.saveAllCurves(self.image_initial)
        self.findTurningPoints2()
        if self.findControlPoints() == False:
            return False
        tp = []
        cp = []
        for i in range(len(self.TurningPoints)):
            self.TurningPoints[i].append(0)
            for j in range(1, len(self.TurningPoints[i])):
                tp.append(str(self.Curves[i][self.TurningPoints[i][j-1]]))
                tp.append(str(self.Curves[i][self.TurningPoints[i][j]]))
            if i == len(self.TurningPoints) - 1:
                break
            tp.append('-1 -1')
            tp.append('-1 -1')
        for i in range(len(self.ControlPoints)):
            for j in range(len(self.ControlPoints[i])):
                cp.append(str(self.ControlPoints[i][j]))
        turnpoint = ';'.join(tp)
        ctrlpoint = ';'.join(cp)
        return turnpoint, ctrlpoint


    def get_initial_outline(self):
        WHITE = self.WHITE
        BLACK = self.BLACK
        image_initial = self.image_initial
        image_copy1 = image_initial.copy()
        image_result = image_initial.copy()

        row_num = image_initial.shape[1]  # 图像的宽
        col_num = image_initial.shape[0]  # 图像的长

        for row in range(row_num):
            for col in range(col_num):
                image_result.itemset((col, row), WHITE)

        for col in range(col_num - 1):
            for row in range(row_num - 1):
                if image_initial.item(col, row) == BLACK:
                    if ((image_initial.item(col, row - 1) == WHITE and image_initial.item(col,
                                                                                        row + 1) == WHITE and image_initial.item(
                            col - 1, row - 1) == WHITE and image_initial.item(col - 1, row) == WHITE and image_initial.item(
                            col - 1, row + 1) == WHITE) and (
                                image_initial.item(col + 1, row - 1) == BLACK and image_initial.item(col + 1,
                                                                                                    row) == BLACK and image_initial.item(
                                col + 1, row + 1) == BLACK)) or \
                            ((image_initial.item(col, row - 1) == WHITE and image_initial.item(col,
                                                                                            row + 1) == WHITE and image_initial.item(
                                col + 1, row - 1) == WHITE and image_initial.item(col + 1,
                                                                                row) == WHITE and image_initial.item(
                                col + 1, row + 1) == WHITE) and (
                                    image_initial.item(col - 1, row - 1) == BLACK and image_initial.item(col - 1,
                                                                                                        row) == BLACK and image_initial.item(
                                col - 1, row + 1) == BLACK)) or \
                            ((image_initial.item(col - 1, row) == WHITE and image_initial.item(col + 1,
                                                                                            row) == WHITE and image_initial.item(
                                col - 1, row + 1) == WHITE and image_initial.item(col,
                                                                                row + 1) == WHITE and image_initial.item(
                                col + 1, row + 1) == WHITE) and (
                                    image_initial.item(col - 1, row - 1) == BLACK and image_initial.item(col,
                                                                                                        row - 1) == BLACK and image_initial.item(
                                col + 1, row - 1) == BLACK)) or \
                            ((image_initial.item(col - 1, row) == WHITE and image_initial.item(col + 1,
                                                                                            row) == WHITE and image_initial.item(
                                col - 1, row - 1) == WHITE and image_initial.item(col,
                                                                                row - 1) == WHITE and image_initial.item(
                                col + 1, row - 1) == WHITE) and (
                                    image_initial.item(col - 1, row + 1) == BLACK and image_initial.item(col,
                                                                                                        row + 1) == BLACK and image_initial.item(
                                col + 1, row + 1) == BLACK)) or \
                            (image_initial.item(col - 1, row - 1) == WHITE and image_initial.item(col - 1,
                                                                                                row) == WHITE and image_initial.item(
                                col - 1, row + 1) == WHITE and image_initial.item(col,
                                                                                row - 1) == WHITE and image_initial.item(
                                col - 1, row) == WHITE and image_initial.item(col - 1,
                                                                            row + 1) == WHITE and image_initial.item(
                                col + 1, row - 1) == WHITE and image_initial.item(col + 1,
                                                                                row) == WHITE and image_initial.item(
                                col + 1, row + 1) == WHITE) or \
                            (image_initial.item(col - 1, row - 1) == BLACK and image_initial.item(col - 1,
                                                                                                row) == WHITE and image_initial.item(
                                col - 1, row + 1) == WHITE and image_initial.item(col,
                                                                                row - 1) == WHITE and image_initial.item(
                                col, row + 1) == WHITE and image_initial.item(col + 1,
                                                                            row - 1) == WHITE and image_initial.item(
                                col + 1, row) == WHITE and image_initial.item(col + 1, row + 1) == WHITE) or \
                            (image_initial.item(col - 1, row - 1) == WHITE and image_initial.item(col - 1,
                                                                                                row) == WHITE and image_initial.item(
                                col - 1, row + 1) == BLACK and image_initial.item(col,
                                                                                row - 1) == WHITE and image_initial.item(
                                col, row + 1) == WHITE and image_initial.item(col + 1,
                                                                            row - 1) == WHITE and image_initial.item(
                                col + 1, row) == WHITE and image_initial.item(col + 1, row + 1) == WHITE) or \
                            (image_initial.item(col - 1, row - 1) == WHITE and image_initial.item(col - 1,
                                                                                                row) == WHITE and image_initial.item(
                                col - 1, row + 1) == WHITE and image_initial.item(col,
                                                                                row - 1) == WHITE and image_initial.item(
                                col, row + 1) == BLACK and image_initial.item(col + 1,
                                                                            row - 1) == WHITE and image_initial.item(
                                col + 1, row) == WHITE and image_initial.item(col + 1, row + 1) == WHITE) or \
                            (image_initial.item(col - 1, row - 1) == WHITE and image_initial.item(col - 1,
                                                                                                row) == WHITE and image_initial.item(
                                col - 1, row + 1) == WHITE and image_initial.item(col,
                                                                                row - 1) == WHITE and image_initial.item(
                                col, row + 1) == WHITE and image_initial.item(col + 1,
                                                                            row - 1) == WHITE and image_initial.item(
                                col + 1, row) == WHITE and image_initial.item(col + 1, row + 1) == BLACK):
                        image_copy1.itemset((col, row), WHITE)
                    else:
                        image_copy1.itemset((col, row), BLACK)
                else:
                    image_copy1.itemset((col, row), WHITE)
        image_result = image_copy1.copy()
        for col in range(col_num - 1):
            for row in range(row_num - 1):
                if image_copy1.item(col, row) == WHITE:
                    if (image_copy1.item(col - 1, row - 1) == BLACK and image_copy1.item(col - 1,
                                                                                        row) == BLACK and image_copy1.item(
                            col, row - 1) == BLACK) or \
                            (image_copy1.item(col - 1, row) == BLACK and image_copy1.item(col - 1,
                                                                                        row + 1) == BLACK and image_copy1.item(
                                col, row + 1) == BLACK) or \
                            (image_copy1.item(col, row - 1) == BLACK and image_copy1.item(col + 1,
                                                                                        row - 1) == BLACK and image_copy1.item(
                                col + 1, row) == BLACK) or \
                            (image_copy1.item(col + 1, row) == BLACK and image_copy1.item(col + 1,
                                                                                        row + 1) == BLACK and image_copy1.item(
                                col, row + 1) == BLACK):
                        image_result.itemset((col, row), BLACK)
                    else:
                        image_result.itemset((col, row), WHITE)
        return image_result


    def get_outline(self):
        WHITE = self.WHITE
        BLACK = self.BLACK
        image = self.get_initial_outline()
        image_copy1 = image.copy()
        image_copy2 = image.copy()
        image_result = image.copy()

        row_num = image.shape[1]  # 图像的宽
        col_num = image.shape[0]  # 图像的长

        for row in range(row_num):
            for col in range(col_num):
                image_result.itemset((col, row), WHITE)

        black_time = 1  # 定义一个连续黑点的次数

        for row in range(row_num - 1):
            for col in range(col_num - 1):
                if image.item(col, row) == WHITE and image.item(col + 1, row) == WHITE:
                    image_copy1.itemset((col, row), WHITE)
                    black_time = 1
                elif image.item(col, row) == WHITE and image.item(col + 1, row) == BLACK:
                    image_copy1.itemset((col, row), WHITE)
                    black_time = 1
                elif image.item(col, row) == BLACK and image.item(col + 1, row) == BLACK:
                    if black_time == 1:
                        image_copy1.itemset((col, row), BLACK)
                        black_time = black_time + 1
                    else:
                        image_copy1.itemset((col, row), WHITE)
                else:
                    image_copy1.itemset((col, row), BLACK)
                    black_time = 1

        for col in range(col_num - 1):
            for row in range(row_num - 1):
                if image.item(col, row) == WHITE and image.item(col, row + 1) == WHITE:
                    image_copy2.itemset((col, row), WHITE)
                    black_time = 1
                elif image.item(col, row) == WHITE and image.item(col, row + 1) == BLACK:
                    image_copy2.itemset((col, row), WHITE)
                    black_time = 1
                elif image.item(col, row) == BLACK and image.item(col, row + 1) == BLACK:
                    if black_time == 1:
                        image_copy2.itemset((col, row), BLACK)
                        black_time = black_time + 1
                    else:
                        image_copy2.itemset((col, row), WHITE)
                else:
                    image_copy2.itemset((col, row), BLACK)
                    black_time = 1

        for row in range(row_num):
            for col in range(col_num):
                if image_copy1.item(col, row) == BLACK or image_copy2.item(col, row) == BLACK:
                    image_result.itemset((col, row), BLACK)
        return image_result




##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
class OneImgProcess:
    #预设变量
    WHITE = 255
    BLACK = 0
    CUT_ROWS = 10  # 切分的行数
    CUT_COLS = 10  # 切分的列数
    BORDER = 30  # 边框宽度
    INIT_P = 65  # 初始预计阈值，黑色像素点为0
    LIMIT_LINE = 30
    cubeSize=500


    goodwidth = 3960  # 正向 / 逆向图像的图像尺寸
    goodheight = 4680
    goodSize = (goodwidth, goodheight)

    imgBinary_ok = 0  # 二值状态
    WORKRUN = 0  # 是否开始工作，默认为0不工作
    CUT_MARGIN = [130, 4520, 100, 3900]  # 切除外边框

    VERTEX = {
        # 四个顶点
        'firstPoint': [142, 612],
        'secondPoint': [2783, 513],
        'thirdPoint': [122, 3706],
        'fourthPoint': [2885, 3746],

    }

    #开始函数
    def __init__(self,readPath,savePath):
        origImg = self.imgGet(readPath)
        picBinary = self.imgBinary(origImg)
        #得到二值图
        self.getPoint(picBinary, 1)
        picPerspective = self.imgTransform(picBinary, self.VERTEX)
        #映射变换
        # pic_direction = self.imgDirection(picPerspective)
        # pic_rotate = self.imgRotate(picPerspective, pic_direction)
        # #完成旋转

        picNoMargin = self.cutImage(picPerspective, self.CUT_MARGIN)
        #完成边框切割
        page_test = [0 for _ in range(11)]
        page,page_level=self.getPage(picNoMargin, page_test, page=0)
        #获得页数

        NoPageCubeIMG=self.getRidOfPageCube(picNoMargin)
        #去除页码块

        noLineImg,HrztCrestList,vtclCrestList =self.clearLine(NoPageCubeIMG)
        #完成抹白表格线
        self.cutSubImg_Save(savePath,page,page_level,noLineImg,HrztCrestList,vtclCrestList)
        #完成切分子图
        pass

        # 问题：
        # √ 1.切割子图时候保存时加上字的信息，字的编号
        #   2.切割外框的准确度
        #   3.计算四点的速度太慢
    #save
    def imgSave(self,savepath, pic):
        if cv2.imwrite(savepath, pic):
            print("save is ok ==>{}".format(savepath))
            return 1
        else:
            print("save error X-X-X{}".format(savepath))
            return 0
    # 去除页码识别块
    def getRidOfPageCube(self,pic):
        height=pic.shape[0]
        img=pic[0:height-self.cubeSize,:]
        savePath=r'getRidOfPageCube.jpg'
        self.imgSave(savePath,img)
        return img


    # 打印功能，显示时间
    def showWork(self,work):
        time_now = datetime.datetime.now().strftime('%H:%M:%S')
        print(time_now + '==>' + work)

    # 读取图像
    def imgGet(self,picpath):
        try:
            pic = cv2.imdecode(np.fromfile(picpath, dtype=np.uint8), 1)
            # pic = cv2.imread(picpath, 1)
            self.showWork("获取图像")
            return pic
        except Exception as e:
            print(e)
    # 返回二值化图像
    def imgBinary(self,pic_copy):
        # pic_copy = pic.copy()
        picGray = cv2.cvtColor(pic_copy, cv2.COLOR_BGR2GRAY)
        # ret, th1 = cv2.threshold(picGray, 127, 255, cv2.THRESH_BINARY)
        ret, th2 = cv2.threshold(picGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大律法二值化
        # imgBinary_ok = 1
        savepath=r'./Binery.jpg'
        self.imgSave(savepath,th2)
        return th2

    # 获取图像信息，返回picInfo字典（长，宽，通道，像素点）
    def getInfo(self,pic):
        pic_copy = pic.copy()

        picInfo = {
            'height': 0,  # 长
            'width': 0,  # 宽
            'channels': 1,  # 通道数,默认为单通道（二值化后）
            'pixnum': 0,  # 像素点数
        }
        try:
            picInfo['height'] = pic_copy.shape[0]
            picInfo['width'] = pic_copy.shape[1]
            if len(pic_copy.shape) > 2:
                picInfo['channels'] = pic_copy.shape[2]
            picInfo['pixnum'] = pic_copy.size
            # showWork('获取图像基本信息')
            # print("图像长度：" + str(picInfo['height']))
            # print("图像宽度：" + str(picInfo['width']))
            # print("像素总数：" + str(picInfo['pixnum']))
            return picInfo
        except Exception as e:
            print(e)

    # 获取表格四个顶点位置，并存到VERTEX中
    def getPoint(self,pic, WORKRUN):
        # if imgBinary_ok == 0:
        #     print("未进行二值化处理，已自动执行。")
        #     imgBinary(pic)
        # showWork("获取页面外框的四个顶点")
        pic_copy = pic.copy()

        picinfo = self.getInfo(pic_copy)

        flag_ok = 0  # 记录扫描是否成功
        bar = 10  # 扫描区域
        limit = self.LIMIT_LINE  # 扫描的阈值

        if WORKRUN == 1:
            # 左上
            while bar < (picinfo['width'] - 1):
                for i in range(0, bar):  # 正向扫描copy图，找到外框线的两个上交点
                    for j in range(0, bar):
                        ii = i
                        jj = j
                        if pic_copy[i][j] == self.BLACK:
                            k = 0
                            # 考虑到有噪点存在，需要 进一步判断, 判断是否有至少20个像素点是连续的
                            while k <= limit and pic_copy[i][jj] == self.BLACK and \
                                    pic_copy[ii][j] == self.BLACK:
                                k += 1
                                ii += 1
                                jj += 1
                            if k >= limit:
                                self.VERTEX['firstPoint'][0] = j
                                self.VERTEX['firstPoint'][1] = i
                                flag_ok = 1
                                print("左上角：")
                                print(str(j) + ',' + str(i))
                                break

                    if flag_ok == 1:
                        break

                if flag_ok == 1:
                    break
                bar += 10

            # 右上
            flag_ok = 0
            bar = 5
            # i = picinfo['width'] - 1
            while bar < (picinfo['width'] - 1):
                for j in range(picinfo['width'] - bar, picinfo['width'])[::-1]:
                    for i in range(0, bar):
                        if pic_copy[i][j] == self.BLACK:
                            # fuck = pic_copy[i][j]
                            ii = i
                            jj = j
                            # cv2.circle(pic_copy, (i, j), 100, (0, 0, 255), 8)
                            k = 0
                            while k <= limit and pic_copy[i][jj] == self.BLACK and \
                                    pic_copy[ii][j] == self.BLACK:
                                k += 1
                                ii += 1
                                jj -= 1
                            if k >= limit:
                                self.VERTEX['secondPoint'][0] = j
                                self.VERTEX['secondPoint'][1] = i
                                flag_ok = 1
                                print("右上角：")
                                print(str(i) + ',' + str(j))
                                break
                    if flag_ok == 1:
                        break
                if flag_ok == 1:
                    break
                bar += 10

            # 左下
            flag_ok = 0
            bar = 10
            while bar < (picinfo['width'] - 1):
                j = picinfo['height'] - 1
                while j > (picinfo['height'] - bar):
                    for i in range(0, bar):
                        ii = i
                        jj = j
                        if pic_copy[j][i] == self.BLACK:
                            k = 0
                            # 考虑到有噪点存在，需要 进一步判断, 判断是否有至少20个像素点是连续的
                            while k <= limit and pic_copy[jj][i] == self.BLACK and \
                                    pic_copy[j][ii] == self.BLACK:
                                k += 1
                                ii += 1
                                jj -= 1
                            if k >= limit:
                                self.VERTEX['thirdPoint'][0] = i
                                self.VERTEX['thirdPoint'][1] = j
                                flag_ok = 1
                                print("左下角：")
                                print(str(i) + ',' + str(j))
                                break
                    j -= 1
                    if flag_ok == 1:
                        break
                if flag_ok == 1:
                    break
                bar += 10

            # 右下
            flag_ok = 0
            bar = 10
            while bar < (picinfo['width'] - 1):
                j = picinfo['height'] - 1
                while j > (picinfo['height'] - bar):
                    i = picinfo['width'] - 1
                    while i > (picinfo['width'] - bar):
                        ii = i
                        jj = j
                        if pic_copy[j][i] == self.BLACK:
                            k = 0
                            # 考虑到有噪点存在，需要 进一步判断, 判断是否有至少20个像素点是连续的
                            while k <= limit and pic_copy[jj][i] == self.BLACK and \
                                    pic_copy[j][ii] == self.BLACK:
                                k += 1
                                ii -= 1
                                jj -= 1
                            if k >= limit:
                                self.VERTEX['fourthPoint'][0] = i
                                self.VERTEX['fourthPoint'][1] = j
                                flag_ok = 1
                                print("右下角：")
                                print(str(i) + ',' + str(j))
                                break
                        i -= 1
                    j -= 1
                    if flag_ok == 1:
                        break
                if flag_ok == 1:
                    break
                bar += 10

            # 标出四个点

            # self.VERTEX['firstPoint'] = tuple(VERTEX['firstPoint'])
            # self.VERTEX['secondPoint'] = tuple(VERTEX['secondPoint'])
            # self.VERTEX['thirdPoint'] = tuple(VERTEX['thirdPoint'])
            # self.VERTEX['fourthPoint'] = tuple(VERTEX['fourthPoint'])

            # cv2.circle(pic_copy, VERTEX['firstPoint'], 100, (0, 0, 255), 15)
            # cv2.circle(pic_copy, VERTEX['secondPoint'], 100, (0, 0, 255), 15)
            # cv2.circle(pic_copy, VERTEX['thirdPoint'], 100, (0, 0, 255), 15)
            # cv2.circle(pic_copy, VERTEX['fourthPoint'], 100, (0, 0, 255), 15)

            return pic_copy

        # imwrite("D:\\ourproject\\bmp\\12.bmp", copy)
        #

    # 返回图像的透视变换后的图
    def imgTransform(self,pic, points):
        self.showWork("透视变换")
        pic_copy = pic.copy()
        picInfo = self.getInfo(pic_copy)
        picWidth = picInfo['width']
        picHeight = picInfo['height']

        # 判断方向
        if picWidth < picHeight:
            transImage = np.zeros((self.goodheight, self.goodwidth), np.float32)

        else:
            transImage = np.zeros((self.goodwidth, self.goodheight), np.float32)
        transInfo = self.getInfo(transImage)
        transWidth = transInfo['width']
        transHeight = transInfo['height']

        # print('透视变换用的顶点：' + str(points))

        pts1 = np.float32([points['firstPoint'], points['secondPoint'],
                           points['thirdPoint'], points['fourthPoint']])
        pts2 = np.float32([[0, 0], [transWidth, 0], [0, transHeight],
                           [transWidth, transHeight], ])

        # 生成透视变换矩阵；进行透视变换
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(pic_copy, M, (transWidth, transHeight))

        savepath=r'./imgTransform.jpg'
        self.imgSave(savepath,dst)
        return dst

    # 图像底部黑像素点扫描，输入透视变换后的图以及子图宽度、高度，返回是否查找黑块成功
    def balckBlaockOk(self,pic_trans, scanway=1):
        pic_trans_info = self.getInfo(pic_trans)
        pic_trans_copy = pic_trans.copy()

        # 单元格尺寸
        ceilwidth = 340
        ceilheight = 340

        startX_rl = int(ceilwidth / 2)  # 左向右扫描开始的横坐标
        startY_rl = pic_trans_info['height'] - ceilheight  # 左向右扫描开始的纵坐标

        startX_tb = ceilwidth  # 上向下扫描开始的横坐标
        startY_tb = int(ceilheight / 2)
        black_px = 0  # 黑像素点数目
        white_px = 0  # 白像素点数目
        flag = 0  # 黑块查找成功标志
        pageblock_ok = 300  # 扫描黑像素数大于等于这个数，代表这是一个页码块

        # 最多有10个黑块
        # 从左向右扫描,用于识别正向还是翻转
        j = 0
        if scanway == 1:
            while j < 11:
                k = 0
                while k < ceilwidth and j * ceilwidth + startX_rl < pic_trans_info['width']:
                    if pic_trans_copy[startY_rl][j * ceilwidth + startX_rl + k] == self.BLACK \
                            and pic_trans_copy[startY_rl + 25][j * ceilwidth + startX_rl + k] == self.BLACK \
                            and pic_trans_copy[startY_rl - 25][j * ceilwidth + startX_rl + k] == self.BLACK:
                        black_px += 1
                    k += 1
                if black_px >= pageblock_ok:
                    flag = 1
                    return 1
                j += 1
                black_px = 0
        else:
            # 从上向下扫描，用于识别左向还是右向
            while j < 11:
                k = 0
                while k < ceilwidth and k <= pic_trans_info['height'] \
                        and j * ceilwidth + startY_tb < pic_trans_info['height']:
                    if pic_trans_copy[startX_tb][j * ceilwidth + startY_tb + k] == self.BLACK \
                            and pic_trans_copy[startX_tb + 25][j * ceilwidth + startY_tb + k] == self.BLACK \
                            and pic_trans_copy[startX_tb - 25][j * ceilwidth + startY_tb + k] == self.BLACK:
                        black_px += 1
                    k += 1
                if black_px >= pageblock_ok:
                    flag = 1
                    return 1
                j += 1
                black_px = 0

        # 未查找到页码块
        if flag == 0:
            return 0

    # 判断页面的方向
    def imgDirection(self,pic):
        tem_pic = pic.copy()
        self.showWork('判定页面方向')
        flag = 0  # 记录扫描状态，成功1、失败0
        pic_info = self.getInfo(tem_pic)

        if pic_info['height'] > pic_info['width']:
            # 判定正向还是逆向
            scan_black = self.balckBlaockOk(tem_pic, scanway=1)
            if scan_black == 1:
                print('正向')
                return 1
            else:
                print('逆向')
                return 2

        else:
            # 判定左向还是右向
            scan_black = self.balckBlaockOk(tem_pic, scanway=0)
            if scan_black == 1:
                print('右倒')
                return 3
            else:
                print('左倒')
                return 4

    # 旋转页面到正确方向#
    def imgRotate(self,pic, direction):
        self.showWork('矫正图像方向')
        rotate_pic = pic

        if direction == 1:  # 正向，无须矫正
            print('图像正向，无需调整')

        elif direction == 2:  # 逆向，逆时针180°
            rotate_pic = cv2.flip(rotate_pic, 0)
            rotate_pic = cv2.flip(rotate_pic, 1)
            print('图像逆向，已调整')
        elif direction == 3:  # 右倒，逆时针旋转90°
            trans_img = cv2.transpose(rotate_pic)
            rotate_pic = cv2.flip(trans_img, 0)
            print('图像右倒，已调整')
        elif direction == 4:  # 左倒，顺时针旋转90°
            trans_img = cv2.transpose(rotate_pic)
            rotate_pic = cv2.flip(trans_img, 1)
            print('图像左倒，已调整')
        else:
            print('图像方向判断出错')
            return 0
        savepath=r'./imgRotate.jpg'
        self.imgSave(savepath,rotate_pic)
        return rotate_pic

    # 去除外框=往里切除size个像素深度
    def cutImage(self,pic, size):
        self.showWork('去除外框')
        col_top = size[0]
        col_bot = size[1]
        row_left = size[2]
        row_right = size[3]

        newimg = pic[col_top:col_bot, row_left:row_right]  # 剪切操作

        savepath=r'./cutImage.jpg'
        self.imgSave(savepath,newimg)
        return newimg

    # 扫描图像页码数#
    def getPage(self,pic_trans, page_arr, page):
        self.showWork("获取页码")

        pic_trans_info = self.getInfo(pic_trans)
        pic_trans_copy = pic_trans.copy()

        ceilimagWidth = int(pic_trans_info['width'] / 10)
        ceilimageHeight = int(pic_trans_info['height'] / 12)

        startX = 0  # 扫描开始的位置
        startY = pic_trans_info['height'] - ceilimageHeight + 250

        flag = 0  # 记录扫描状态，成功1、失败0
        pageblock_ok = 100  # 扫描黑像素数大于等于这个数，代表这是一个页码块
        black_px = 0
        page_level = 1

        print('页码二进制表示：【', end='')
        j = 0
        while j < 11:
            # 每隔330个，再加30步长进行一次连续扫描
            k = 0
            while k < ceilimagWidth and j * ceilimagWidth + startX < pic_trans_info['width']:
                picX = j * ceilimagWidth + startX + k
                picY = startY
                if pic_trans_copy[picY][picX] == self.BLACK:
                    black_px += 1
                k += 1
            if black_px >= pageblock_ok and j + 1 <= len(page_arr):
                page_arr[j + 1] = 1
                page_arr[0] = 1  # 第一个位置用于记录查找状态
                flag = 1
            if j + 1 < 7 and j + 1 <= len(page_arr):
                print(page_arr[j + 1], end='')
                if page_arr[j + 1] == 1:
                    page += pow(2, j)  # 二进制转十进制
            # 识别第8，9，10个方块，确定页面汉字级别，

            if (page_arr[9] == 0 and page_arr[8] == 0 and page_arr[7] == 1):
                page_level = 3  # 标点符号页
            if (page_arr[9] == 0 and page_arr[8] == 1 and page_arr[7] == 0):
                page_level = 2  # 二级汉字页
            if (page_arr[9] == 0 and page_arr[8] == 0 and page_arr[7] == 0):
                page_level = 1  # 一级汉字页
            black_px = 0
            j += 1
        printstring = "】 -----第 " + str(page) + " 页," + str(page_level) + "级汉字\n"
        print(printstring)
        return page,page_level

    # 画出投影图
    def drawProjection(self,projectLs,fullPix,dir):
        if dir=="Horizontal":
            sz=len(projectLs)
            img=np.full((sz,fullPix),self.WHITE,int)
            for rowIdx in range(0,sz):
                for colIdx in range(0,projectLs[rowIdx]):
                    img[rowIdx][colIdx]=self.BLACK
            
            self.imgSave(r"testpic/Horizontal.jpg",img)

        elif dir=="vertical":
            sz=len(projectLs)
            img=np.full( (fullPix,sz),self.WHITE,int)
            for colIdx in range(0,sz):
                for rowIdx in range(0,projectLs[colIdx]):
                    img[rowIdx][colIdx]=self.BLACK
            self.imgSave(r"testpic/vertical.jpg",img)
        else:
            print("投影方向出错")
            return

# 涂白表格线并返回
    def clearLine(self,pic): #传入一个图，返回一个去除所有，只剩汉字的图片

        height, width = pic.shape[:2]
        # 水平投影
        Horizontal=list()
        szOfRow=len(pic[0])
        # sumINRow=0
        for row in pic:
            Color_B=szOfRow-int(cv2.countNonZero(row))
            Horizontal.append(Color_B)
            # sumINRow+=Color_B
        # avgRow=int(sumINRow/height)


        # 垂直投影
        vertical=list()
        szOfCol=len(pic[:,0])
        # sumINCol=0
        for colIdx in range(0,width):
            col=pic[:,colIdx]
            Color_B=szOfCol-int(cv2.countNonZero(col))
            vertical.append(Color_B)
            # sumINCol+=Color_B
        # avgCol=int(sumINCol/width)


        # # #画出投影图
        # Himg=self.drawProjection(Horizontal,len(vertical),"Horizontal","Horizontal")
        # Vimg=self.drawProjection(vertical,len(Horizontal),"vertical","vertical")
        



        # 获取 水平投影和垂直投影的波峰 波谷
        HrztCrestList,HrztTroughList=self.crestTrough(Horizontal)
        vtclCrestList,vtclTroughList=self.crestTrough(vertical)

        ##### 抹白
        #按行抹白
        #第一行到第一个波谷
        color_WHITE=(255,255,255)
        for rowIdx in range(0,HrztTroughList[0]+1):
            cv2.line(pic,(0,rowIdx),(width,rowIdx),color_WHITE)
        #中间的波谷到波谷
        troughIdx=1
        for i in range(0,9):
            rowSTART=HrztTroughList[troughIdx]
            rowEND=HrztTroughList[troughIdx+1]
            for rowIdx in range(rowSTART,rowEND+1):
                cv2.line(pic,(0,rowIdx),(width,rowIdx),color_WHITE)
            troughIdx+=2        
        #最后一个波谷到最后一行
        for rowIdx in range(HrztTroughList[-1],height+1):
            cv2.line(pic,(0,rowIdx),(width,rowIdx),color_WHITE)

        #按列抹白
        #第一列到第一个波谷
        for colIdx in range(0,vtclTroughList[0]+1):
            cv2.line(pic,(colIdx,0),(colIdx,width),color_WHITE)
        #中间的波谷到波谷
        troughIdx=1
        for i in range(0,9):
            rowSTART=vtclTroughList[troughIdx]
            rowEND=vtclTroughList[troughIdx+1]
            for colIdx in range(rowSTART,rowEND+1):
                cv2.line(pic,(colIdx,0),(colIdx,width),color_WHITE)
            troughIdx+=2        
        #最后一个波谷到最后一列
        for colIdx in range(vtclTroughList[-1],height+1):
            cv2.line(pic,(colIdx,0),(colIdx,width),color_WHITE)

        self.imgSave("抹白.jpg",pic)
        return pic,HrztCrestList,vtclCrestList



#（协助涂白表格线）找到投影图的波峰和波谷，并返回波峰波谷的index列表
    def crestTrough(self,ls):        
        sz=len(ls)
        crestList=list()    #波峰 11
        troughList=list()   #波谷 20
         # 波峰---寻找超过 fullpix/2 的所有点

        sz=len(ls)
        onePiece= int(sz/11)
        for i in range(0,11):
            startPix=i*onePiece
            endPix=startPix+onePiece
            nowPiece=ls[startPix:endPix]
            m=max(nowPiece)
            Idx=nowPiece.index(m)+startPix
            crestList.append(Idx)
            


        #波谷---寻找最小值，如果寻找过程中，当前值上涨超出上一个最小值的的一半，判定为当前极值寻找结束，minNum=fullPix，开始寻找下一个
        minNum=ls[crestList[0]]
        # pltLowList=np.full(sz,0,int)
        for i in range(crestList[0],sz):
            if ls[i]<minNum:
                minIdx=i
                minNum=ls[i]
            if ls[i]-minNum>(minNum*0.4):
                break
        sameDistance=minIdx-crestList[0]

        for cre in crestList:
            if cre != crestList[0]:
                troughList.append(cre-sameDistance)
                # pltLowList[cre-sameDistance]=ls[cre-sameDistance]
            if cre != crestList[-1]:
                troughList.append(cre+sameDistance)
        #         pltLowList[cre+sameDistance]=ls[cre+sameDistance]
        # plt.plot(range(0,sz), ls,label='polyfit values')
        # # plt.plot(range(0,sz), pltHighList,'*',label='polyfit values')
        # plt.plot(range(0,sz), pltLowList,'+',label='polyfit values')
        # plt.show()
        # print("")
        return crestList,troughList

    #切割子图并保存
    def cutSubImg_Save(self,savePath,page,page_level,noLinePic,HrztCrestList,vtclCrestList):
        height, width = noLinePic.shape[:2]

        for rowBotIdx in range(1,len(HrztCrestList)):
            rowBot=HrztCrestList[rowBotIdx]
            rowTop=HrztCrestList[rowBotIdx-1]
            thisRowPriodPic=noLinePic[rowTop:rowBot+1]
            for colBotIdx in range(1,len(vtclCrestList)):
                colBot=vtclCrestList[colBotIdx]
                colTop=vtclCrestList[colBotIdx-1]
                subPic=thisRowPriodPic[:,colTop:colBot]
                wordId=(page-1)*100+(rowBotIdx-1)*10+colBotIdx
                subSavePath=savePath+"/"+str(wordId)+'_'+str(page)+'_'+str(rowBotIdx)+'_'+str(colBotIdx)+".jpg"
                self.imgSave(subSavePath,subPic)



# if __name__=="__main__":

#     readPath= r"testpic/best_test.jpg"
#     savePath= r"start/save"
#     test=OneImgProcess(readPath,savePath)