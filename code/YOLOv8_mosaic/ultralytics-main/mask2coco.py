import json
import numpy as np
from pycocotools import mask
import cv2
import os
import sys

if sys.version_info[0] >= 3:
    unicode = str

import io

# 实例的id，每个图像有多个物体每个物体的唯一id
global segmentation_id
segmentation_id = 1


# annotations部分的实现
def maskToanno(ground_truth_binary_mask, ann_count, category_id):
    contours, _ = cv2.findContours(ground_truth_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 根据二值图找轮廓
    annotations = []  # 一幅图片所有的annotatons
    # print(len(contours),contours)
    global segmentation_id
    if (len(contours) == 0): print("0")
    # 对每个实例进行处理
    for i, contour in enumerate(contours):
        if (len(contour) < 3):
            print("The contour does not constitute an area")
            continue
        ground_truth_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        annotation = {
            "segmentation": [],
            "area": ground_truth_area,
            "iscrowd": 0,
            "image_id": ann_count,
            "bbox": [x, y, w, h],
            "category_id": category_id,
            "id": segmentation_id
        }
        # 求segmentation部分
        contour = np.flip(contour, axis=0)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
        annotations.append(annotation)
        segmentation_id = segmentation_id + 1
    return annotations


# mask图像路径
block_mask_path = './dataset/CHIP/masks'
block_mask_image_files = sorted(os.listdir(block_mask_path))

# coco json保存的位置
jsonPath = "./dataset/CHIP/json_labels/label.json"
annCount = 1
imageCount = 1
# 原图像的路径， 原图像和mask图像的名称是一致的。
path = "./dataset/CHIP/images"
rgb_image_files = sorted(os.listdir(path))
if block_mask_image_files != rgb_image_files: print("error")

with io.open(jsonPath, 'w', encoding='utf8') as output:
    # 那就全部写在一个文件夹好了
    output.write(unicode('{\n'))
    # 基本信息
    output.write(unicode('"info": [\n'))
    output.write(unicode('{\n'))
    info = {
        "year": "2023",
        "version": "1",
        "contributor": "bulibuli",
        "url": "",
        "date_created": "2023-01-17"
    }
    str_ = json.dumps(info, indent=4)
    str_ = str_[1:-1]
    if len(str_) > 0:
        output.write(unicode(str_))
    output.write(unicode('}\n'))
    output.write(unicode('],\n'))

    # lisence
    output.write(unicode('"lisence": [\n'))
    output.write(unicode('{\n'))
    info = {
        "id": 1,
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "name": "CC BY 4.0"
    }
    str_ = json.dumps(info, indent=4)
    str_ = str_[1:-1]
    if len(str_) > 0:
        output.write(unicode(str_))
    output.write(unicode('}\n'))
    output.write(unicode('],\n'))

    # category
    output.write(unicode('"categories": [\n'))
    output.write(unicode('{\n'))
    categories = [
        {
            "supercategory": "Clothes",
            "id": 1,
            "name": "Hat"
        },
        {
            "supercategory": "Body-parts",
            "id": 2,
            "name": "Hair"
        },
        {
            "supercategory": "Clothes",
            "id": 3,
            "name": "Glove"
        },
        {
            "supercategory": "Clothes",
            "id": 4,
            "name": "Sunglasses"
        },
        {
            "supercategory": "Clothes",
            "id": 5,
            "name": "UpperClothes"
        },
        {
            "supercategory": "Clothes",
            "id": 6,
            "name": "Dress"
        },
        {
            "supercategory": "Clothes",
            "id": 7,
            "name": "Coat"
        },
        {
            "supercategory": "Clothes",
            "id": 8,
            "name": "Socks"
        },
        {
            "supercategory": "Clothes",
            "id": 9,
            "name": "Pants"
        },
        {
            "supercategory": "Body-parts",
            "id": 10,
            "name": "Torso-skin"
        },
        {
            "supercategory": "Clothes",
            "id": 11,
            "name": "Scarf"
        },
        {
            "supercategory": "Clothes",
            "id": 12,
            "name": "Skirt"
        },
        {
            "supercategory": "Body-parts",
            "id": 13,
            "name": "Face"
        },
        {
            "supercategory": "Body-parts",
            "id": 14,
            "name": "Left-arm"
        },
        {
            "supercategory": "Body-parts",
            "id": 15,
            "name": "Right-arm"
        },
        {
            "supercategory": "Body-parts",
            "id": 16,
            "name": "Left-leg"
        },
        {
            "supercategory": "Body-parts",
            "id": 17,
            "name": "Right-leg"
        },
        {
            "supercategory": "Clothes",
            "id": 18,
            "name": "Left-shoe"
        },
        {
            "supercategory": "Clothes",
            "id": 19,
            "name": "Right-shoe"
        }
    ]
    str_ = json.dumps(categories, indent=4)
    str_ = str_[1:-1]
    if len(str_) > 0:
        output.write(unicode(str_))
    output.write(unicode('}\n'))
    output.write(unicode('],\n'))

    # images

    output.write(unicode('"images": [\n'))
    for image in rgb_image_files:
        if os.path.exists(os.path.join(block_mask_path, image)):
            output.write(unicode('{'))
            block_im = cv2.imread(os.path.join(path, image))
            h, w, _ = block_im.shape
            annotation = {
                "height": h,
                "width": w,
                "id": imageCount,
                "file_name": image
            }
            str_ = json.dumps(annotation, indent=4)
            str_ = str_[1:-1]
            if len(str_) > 0:
                output.write(unicode(str_))
                imageCount = imageCount + 1
            if (image == rgb_image_files[-1]):
                output.write(unicode('}\n'))
            else:
                output.write(unicode('},\n'))
    output.write(unicode('],\n'))

    # 写annotations
    output.write(unicode('"annotations": [\n'))
    for i in range(len(block_mask_image_files)):
        if os.path.exists(os.path.join(path, block_mask_image_files[i])):
            block_image = block_mask_image_files[i]
            # print(block_image)
            # 读取二值图像
            block_im = cv2.imread(os.path.join(block_mask_path, block_image), 0)
            _, block_im = cv2.threshold(block_im, 100, 1, cv2.THRESH_BINARY)
            if not block_im is None:
                block_im = np.array(block_im, dtype=object).astype(np.uint8)
                block_anno = maskToanno(block_im, annCount, 1)
                # print(block_image,len(block_anno))
                for b in block_anno:
                    str_block = json.dumps(b, indent=4)
                    str_block = str_block[1:-1]
                    if len(str_block) > 0:
                        output.write(unicode('{\n'))
                        output.write(unicode(str_block))
                        if (block_image == rgb_image_files[-1] and b == block_anno[-1]):
                            output.write(unicode('}\n'))
                        else:
                            output.write(unicode('},\n'))
                annCount = annCount + 1
            else:
                print(block_image)

    output.write(unicode(']\n'))
    output.write(unicode('}\n'))

