import os
from PIL import Image


# 获取指定目录下的所有jpg图片
def get_all_jpg_files(dir):
    files_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                files_list.append(os.path.join(root, file))
    return files_list


# 批量转换jpg图片为png格式
def jpg2png(files_list):
    for file in files_list:
        img = Image.open(file)
        new_file = os.path.splitext(file)[0] + '.png'
        img.convert('RGB').save(new_file)


if __name__ == '__main__':
    dir = './dataset/CHIP/images'
    files_list = get_all_jpg_files(dir)
    jpg2png(files_list)