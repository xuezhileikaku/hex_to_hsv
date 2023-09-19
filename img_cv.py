#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project ：djangoProject 
@File    ：tool_cv.py
@IDE     ：PyCharm 
@Author  ：xuezhileikaku
@Date    ：2023/9/13 23:14 
'''
import time

import cv2
import numpy as np


# 定义红色范围
def range_red():
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([13, 255, 255])
    red_ran = {"lower": lower_red, "upper": upper_red}
    return red_ran


# 定义蓝色范围
def range_blue():
    lower_blue = np.array([90, 43, 46])
    upper_blue = np.array([130, 255, 255])
    blue_ran = {"lower": lower_blue, "upper": upper_blue}
    return blue_ran


# 定义黄色范围
def range_yellow():
    lower_yellow = np.array([20, 43, 46])
    upper_yellow = np.array([40, 255, 255])
    yellow_ran = {"lower": lower_yellow, "upper": upper_yellow}
    return yellow_ran


# 定义白色范围
def range_white():
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 43, 255])
    white_ran = {"lower": lower_white, "upper": upper_white}
    return white_ran


# 读取图像
image = cv2.imread('../../static/scan/original/test.jpg')
# image = cv2.imread('../static/scan/original/huan.jpg')
# 方法1#
# 将图像转换为HSV颜色空间
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# lower_blue = np.array([0, 50, 50])
# upper_blue = np.array([10, 255, 255])
# low_up = range_blue()
# print(low_up)
# # 创建掩码
# mask = cv2.inRange(hsv_image, low_up['lower'], low_up['upper'])
#
# # 对原始图像和掩码进行按位与运算，以便只显示特定颜色的区域
# result = cv2.bitwise_and(image, image, mask=mask)
#
# # 将结果图像转换为灰度
# gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#
# # 显示结果
# cv2.imshow('Grayscale Image', gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 保存修改后的灰度图像到本地路径
# output_path = '../static/scan/checked/' + str(time.time()) + '.jpg'  # 替换为你想要的输出路径和文件名
# cv2.imwrite(output_path, gray_image)

# 方法2
# def cv_show(name, img):
#     cv2.imshow(name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# # image = cv2.imread('../static/scan/original/test.jpg')
# low_up = range_blue()
#
# cv_show("img", image)  # 展示原图
# # print(img.shape)
# img = cv2.GaussianBlur(image, (11, 11), 0)  # 高斯滤波降噪，模糊图片
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转成HSV
#
# # img = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)[1]
# color_img = cv2.inRange(img, low_up["lower"], low_up["upper"])  # 筛选出符合的颜色
# kernel = np.ones((3, 3), np.uint8)  # 核定义
# color_img = cv2.erode(color_img, kernel, iterations=2)  # 腐蚀除去相关性小的颜色
# color_img = cv2.GaussianBlur(color_img, (11, 11), 0)  # 模糊图像
# cnts = cv2.findContours(color_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # 找出轮廓
# # print(cnts)
# for cnt in cnts:  # 遍历所有符合的轮廓
#     x, y, w, h = cv2.boundingRect(cnt)
#     # cv2.rectangle(imge, (x, y), (x + w, y + h), (0, 0, 255), 3)
#     cv2.drawContours(image.copy(), cnt, -1, (0, 0, 255), 2)
# cv_show("img", image)  # 展示处理后的图片
# # 保存修改后的灰度图像到本地路径
# output_path = '../static/scan/checked/' + str(time.time()) + '.jpg'  # 替换为你想要的输出路径和文件名
# cv2.imwrite(output_path, image)

# 方法3
# 颜色RBG取值
# 定义HSV阈值
lower_red1 = (150, 43, 46)
upper_red1 = (180, 255, 255)

lower_red2 = (0, 43, 46)
upper_red2 = (13, 255, 255)


def find_red(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 提取深红色部分
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    # 提取浅红色部分
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    # 合并两个掩码
    mask = mask1 + mask2
    return mask


color = {
    "blue": {"color_lower": np.array([100, 43, 46]), "color_upper": np.array([124, 255, 255])},
    "red": {"color_lower": np.array([156, 43, 46]), "color_upper": np.array([180, 255, 255])},
    "yellow": {"color_lower": np.array([26, 43, 46]), "color_upper": np.array([34, 255, 255])},
    "green": {"color_lower": np.array([35, 43, 46]), "color_upper": np.array([77, 255, 255])},
    "purple": {"color_lower": np.array([125, 43, 46]), "color_upper": np.array([155, 255, 255])},
    "orange": {"color_lower": np.array([11, 43, 46]), "color_upper": np.array([25, 255, 255])}
}


def title():
    print("*" * 50 + "颜色识别" + "*" * 50)
    imge = input("请输入你图片的路径：")
    print('识别颜色种类有："blue","red","yellow","green","purple","orange"')
    color_0 = input("请输入你想要识别的颜色：")
    return color_0, imge


def content(color_0, imge):
    imge = cv2.imread(imge)
    # imge = cv2.imread('D:\\Picture\\network\\color.jpg')
    cv_show("img", imge)  # 展示原图
    # print(img.shape)
    img = cv2.cvtColor(imge, cv2.COLOR_BGR2HSV)  # 转成HSV
    img = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波降噪，模糊图片
    # img = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)[1]
    color_img = cv2.inRange(img, color[color_0]["color_lower"], color[color_0]["color_upper"])  # 筛选出符合的颜色
    kernel = np.ones((3, 3), np.uint8)  # 核定义
    color_img = cv2.erode(color_img, kernel, iterations=2)  # 腐蚀除去相关性小的颜色
    color_img = cv2.GaussianBlur(color_img, (5, 5), 0)  # 模糊图像
    cnts = cv2.findContours(color_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # 找出轮廓
    print(cnts)
    for cnt in cnts:  # 遍历所有符合的轮廓
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(imge, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.drawContours(imge.copy(), cnt, -1, (0, 0, 255), 2)
    cv_show("img", imge)  # 展示处理后的图片


def color_find_contours(src_image, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV):
    buf_img = np.zeros_like(src_image)
    img_hsv = np.zeros_like(src_image)
    mask = np.zeros_like(src_image)
    gauss = np.zeros_like(src_image)

    # 高斯滤波
    cv2.GaussianBlur(src_image, (11, 11), 0, gauss)

    # 转为HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV, img_hsv)

    # 提取颜色
    mask = cv2.inRange(img_hsv, (iLowH, iLowS, iLowV), (iHighH, iHighS, iHighV))

    # 开闭运算
    kernel = np.ones((5, 5), np.uint8)
    buf_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    buf_img = cv2.morphologyEx(buf_img, cv2.MORPH_CLOSE, kernel)

    # 显示提取图
    show_plane()
    show_texture(buf_img, select)

    # 翻转图像
    buf_img = cv2.flip(buf_img, 0)

    return buf_img


# color_0, imge= title()
# content(color_0, imge)


def select_red(image):
    img = cv2.imread(image)
    width = img.shape[1]
    height = img.shape[0]
    img_src = cv2.Mat(height, width, cv2.CV_8UC4)
    buf = find_red(image)
    draw_con_line(img_src, buf)


def draw_con_line(img):
    # 创建一个红色的掩码
    mask = cv2.inRange(img, color["blue"]["color_lower"], color["blue"]["color_upper"])
    print(mask);
    # 对原图像和掩码进行位运算
    res = cv2.bitwise_and(img, img, mask=mask)
    print(mask);
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # 找出轮廓
    for cnt in cnts:  # 遍历所有符合的轮廓
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawContours(img.copy(), cnt, -1, (255, 0, 0), 2)



img_file = '../../static/scan/original/test.jpg'


# 读取图像
img = cv2.imread(img_file)
#高斯滤波
gauss = cv2.GaussianBlur(img, (11, 11), 0)
# 转换到HSV空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义红色阈值
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# 提取蓝色部分
mask = cv2.inRange(hsv, color["blue"]["color_lower"], color["blue"]["color_upper"])

# 寻找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上绘制蓝色轮廓
cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示结果
cv2.imshow('Result', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
