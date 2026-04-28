import json
import os
import argparse
from tqdm import tqdm
import glob
import cv2
import numpy as np

def check_labels(txt_labels, images_dir):
    txt_files = glob.glob(txt_labels + "/*.txt")
    for txt_file in txt_files:
        filename = os.path.splitext(os.path.basename(txt_file))[0]

        pic_path = images_dir + filename + ".jpg"

        img = cv2.imread(pic_path)
        height, width, _ = img.shape

        file_handle = open(txt_file)
        cnt_info = file_handle.readlines()
        new_cnt_info = [line_str.replace("\n", "").split(" ") for line_str in cnt_info]

        color_map = {"0": (0, 255, 255)}
        for new_info in new_cnt_info:
            print(new_info)
            s = []
            for i in range(1, len(new_info), 2):
                b = [float(tmp) for tmp in new_info[i:i + 2]]
                s.append([int(b[0] * width), int(b[1] * height)])
            cv2.polylines(img, [np.array(s, np.int32)], True, color_map.get(new_info[0]))
        cv2.namedWindow('img2', 0)
        cv2.imshow('img2', img)
        cv2.waitKey()

if __name__=='__main__':
    images_dir = 'dataset/images'
    txt_labels = '0000002'
    check_labels(txt_labels, images_dir)