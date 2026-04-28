from ultralytics import YOLO
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 图片文件扩展名
image_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

# 文件路径：只适合全图片或者单次视频(摄像头)的情况
# 图像
file_paths = ['./dataset/human_body/images/0001031.jpg']
# 摄像头
# file_paths = '0'

model = YOLO('./runs/segment/CHIP_train/weights/best.pt')

# 提取文件名
file_name = os.path.basename(file_paths[0])
# 是否为图片
is_picture_type = file_name.lower().endswith(image_type)

if is_picture_type:
    result = model.predict(source=file_paths)
    annotated_frame = result[0].plot()
    print(f'result[0]:\n{result[0]}\n')
    print(f'result[0].boxes:\n{result[0].boxes}\n')
    print(f'result[0].masks:\n{result[0].masks}\n')
    cv2.namedWindow("YOLOv8 Inference (picture)", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Inference (picture)", annotated_frame)
    # 按任意键可关闭cv2窗口
    cv2.waitKey()
if file_paths == '0':
    result = model.predict(source=file_paths, show=True)
