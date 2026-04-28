from ultralytics import YOLO
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 图片文件扩展名
image_type = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff','.webp')

# 说明：以下关于颜色的设置均为RGB颜色

# 打码模式1（单色打码模式）的基本色彩
single_color = {'b': [0, 0, 1], 'g': [0, 0.5, 0], 'r': [1, 0, 0], 'c': [0, 0.75, 0.75],
              'm': [0.75, 0, 0.75], 'y': [0.75, 0.75, 0], 'k': [0, 0, 0], 'w': [1, 1, 1]}

# 打码模式2（多色打码模式）的默认掩膜颜色表（16进制）
# 参考了plotting.py中的掩膜颜色表
hexs = ("#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
        "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7")

# 将16进制转为十进制
def hex2rgb(h):
    return list(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

nc = 19     # 数据集的目标种类数
palette = [hex2rgb(c) for c in hexs]      # 掩膜颜色表（十进制）
mask_colors = [palette[i % len(palette)] for i in range(nc)]    # 掩膜颜色


# 需要打码的类别序号（从0开始）
# mosaic_index = [1, 9, 12]  # CHIP和human_body
mosaic_index = [0]   # COCO

# 透明度alpha：0-1
alpha = 1

# 打码模式
mosaic_mode = 2
# 说明：打码模式mosaic_mode取不同值的含义如下
# 0：需要打码的类别均以类似于新闻上的方式打码（通过最临近插值，形成块状马赛克效果）。
# 1：需要打码的类别均打上同一种颜色的码，例如打码区域全白、全红等。
# 2：需要打码的不同类别根据指定类别颜色进行打码。


# mosaic_mode = 0时，设置项：
mosaic_size = 64    # 打码尺寸（图片先缩小时的尺寸）


# mosaic_mode = 1时，设置项：
self_color = False    # 是否使用自己设置的单一颜色：True对应self_color，False对应基本色彩
# 自己设置的单一颜色
# 可以填入RGB构成的列表，如[255, 255, 255]，也可以填入RGB16进制字符串模式（"#??????"），
# 如"#FF0000"（#不要漏，？=0,1,2,...,9,A,B,C,D,E,F）。
# self_mosaic_color = [128, 128, 128]
self_mosaic_color = "#FF1493"
self_mosaic_color = (hex2rgb(self_mosaic_color) if self_mosaic_color[0] == "#" else self_mosaic_color)
single_color_select = "g"       #使用基本色彩
# 打码颜色（单色）
mosaic_color = self_mosaic_color if self_color else [x*255 for x in single_color[single_color_select]]


# mosaic_mode = 2时，设置项：
self_colors = True     # 是否使用自己设置的不同类别的多种颜色：True对应self_colors，False对应掩膜颜色
from_single_color = True        # 自己设置的颜色是否均来自于（单色打码模式）的基本色彩
# 自己设置的不同类别的多种颜色（自己设置的RGB值），其中不同行对应需要打码的类别。
# 每个类别可以填入RGB构成的列表，如[255, 255, 255]，也可以填入RGB16进制字符串模式（"#??????"），
# 如"#FF0000"（#不要漏，？=0,1,2,...,9,A,B,C,D,E,F）。
self_mosaic_colors = [[128, 128, 128],
                      "#FF1493",
                      [0, 255, 255]]
self_mosaic_colors = [(hex2rgb(x) if x[0] == "#" else x) for x in self_mosaic_colors]
# 由基本色彩构成的list（元素为颜色对应字母），用于映射到对应的RGB值
single_color_list = ["g", "r", "c"]
single_color_mosaic_colors = []   # 由基本色彩构成的，自己设置的颜色RGB值
for i in range(len(single_color_list)):
    single_color_mosaic_colors.append([x*255 for x in single_color[single_color_list[i]]])
# 自己设置的不同类别的多种颜色（根据from_single_color选择是否来源于基本色彩还是自己设置的RGB值）
self_mosaic_colors = single_color_mosaic_colors if from_single_color else self_mosaic_colors
# 打码颜色（多色）
mosaic_colors = self_mosaic_colors if self_colors else [mask_colors[x] for x in mosaic_index]


# 保存图片的部分路径
img_dir_begin = './predict_files/predict/coco128-seg/picture'

# 保存视频（录像）的部分路径
video_dir_begin = './predict_files/predict/coco128-seg/video'

# 保存状态
is_save = {'save original image': False,
           'save mosaic image': False,
           'save image of YOLOv8 inference': False,
           'save figure': False,
           'save original video': False,
           'save mosaic video': False,
           'save video of YOLOv8 inference': False}

# 默认保存状态
is_save0 = {'save original image': True,
            'save mosaic image': True,
            'save image of YOLOv8 inference': True,
            'save figure': True,
            'save original video': True,
            'save mosaic video': True,
            'save video of YOLOv8 inference': True}

# 保存状态说明：
# 'save original image': 保存原始图片
# 'save mosaic image': 保存打码图片
# 'save image of YOLOv8 inference': 保存（含打码）图片的YOLOv8分割结果
# 'save figure': 保存打码过程中的所有图片构成的figure窗口（图片形式）
# 'save original video': 保存原始视频（录像）
# 'save mosaic video': 保存打码视频（录像）
# 'save video of YOLOv8 inference': 保存（含打码）视频（录像）的YOLOv8分割结果


def prediction_mosaic(result, orig_img0, draw_mosaic, draw_all, is_picture_type, mosaic_mode=mosaic_mode, not_save_video=False):

    # orig_img0: 原图像
    # result: 未打码的推理结果
    # orig_img: 将原图像的尺寸缩放到训练时的输入尺寸
    # mask: 对所有需要打码的类型对应的区域求并集后的整体掩膜
    # category_mask: 单个打码类别的所有物体对应的掩膜区域求并集
    # keying_masked_image: 原图像抠掉的需要打码的区域（未打码）
    # keying_image: 原图像抠掉需要打码的区域后剩下的区域
    # mosaic_image: 对整个原图像打码后的图像
    # mosaic_masked_image: mosaic_image抠掉的打码区域，相当于对keying_masked_image打码后的结果
    # mosaic: 拼接后的打码图像（没有框和分割区域）
    # annotated_frame: 打码的推理结果

    # 说明：
    # 代码中目前是对矩形区域打码，如果对keying_masked_image打码，后面拼接后可能会出现黑色或者其他颜色的边界裂缝。
    # 故需要先对整个图像打码，再抠图、拼接；而不是先抠图，再对抠图部分打码，再拼接。

    height, width, _ = orig_img0.shape
    train_input_size = result[0].masks.data.shape[-1:0:-1]
    # 这里特别注意，因为使用yolov8训练的时候默认会把图片resize成训练的尺寸，所以这里也得改成你训练的尺寸
    orig_img = cv2.resize(orig_img0, train_input_size)  # 注意OpenCV中尺寸是先宽度后高度

    # -----对所有需要打码的类型对应的区域求并集-----
    # 说明：mask对应的区域是与边界框一一对应（有多少个边界框就有多少个二维mask），且边界框都有对应的类型，故mask中的区域也是有对应的类型。
    mask_index = torch.zeros_like(result[0].boxes.cls)
    for x in range(0, len(mosaic_index)):
        mask_index = mask_index + (result[0].boxes.cls == mosaic_index[x])
    mask = result[0].masks.data[mask_index > 0].cpu().numpy().astype(np.bool_)
    mask = (np.sum(mask, axis=0) > 0)   # 防止打码的区域之间有重叠

    keying_masked_image = np.zeros_like(orig_img).astype('uint8')
    keying_masked_image[mask, :] = orig_img[mask, :]
    # 原图像抠掉需要打码的区域后剩下的区域 = 原图像 - 原图像抠掉的需要打码的区域（未打码）
    keying_image = orig_img - keying_masked_image

    if mosaic_mode == 0:
        # 对矩形区域打码：先缩小，再还原到原来的大小，缩放的方法均为最临近插值。
        mosaic_image = cv2.resize(orig_img, (mosaic_size, mosaic_size), interpolation=cv2.INTER_NEAREST)
        mosaic_image = cv2.resize(mosaic_image, train_input_size, interpolation=cv2.INTER_NEAREST)

        mosaic_masked_image = np.zeros_like(orig_img).astype('uint8')
        mosaic_masked_image[mask, :] = mosaic_image[mask, :]

    if mosaic_mode == 1:
        mosaic_masked_image = keying_masked_image.copy()
        mosaic_masked_image[mask, :] = mosaic_color[::-1]

    if mosaic_mode == 2:
        mosaic_masked_image = keying_masked_image.copy()
        for x in range(0, len(mosaic_index)):
            # 单个打码类别的所有物体对应的掩膜区域求并集
            category_mask = result[0].masks.data[result[0].boxes.cls == mosaic_index[x]].cpu().numpy().astype(np.bool_)
            category_mask = (np.sum(category_mask, axis=0) > 0)  # 防止打码的区域之间有重叠
            mosaic_masked_image[category_mask, :] = mosaic_colors[x][::-1]

    # 拼接：拼接后的打码图像（没有框和分割区域） = mosaic_image抠掉的打码区域*alpha + 原图像抠掉需要打码的区域后剩下的区域
    mosaic = cv2.addWeighted(keying_image, 1, mosaic_masked_image, alpha, 0)
    mosaic = cv2.resize(mosaic, (width, height))

    # 将推理结果中的图像替换为拼接后的打码图像
    result[0].orig_img = mosaic
    annotated_frame = result[0].plot()

    # figure窗口绘图：展示了打码过程中各个阶段对应的图像
    # 说明：视频状态下显示figure窗口会出现间断情况（需要关闭figure窗口才有变化，且变化的为关掉的瞬间对应的图像），保存视频时不会显示figure窗口。
    # 如果处理的是视频且没有保存视频，或者处理的是图片，才可能会显示、保存figure窗口（视频没有设置保存figure窗口选项（原因见上一行的说明））。
    if draw_all and (is_picture_type or (not is_picture_type) and not_save_video):
        plt.figure(num="Mosaic")
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(orig_img0, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        if mosaic_mode == 0:
            plt.subplot(2, 3, 2)
            plt.imshow(cv2.cvtColor(mosaic_image, cv2.COLOR_BGR2RGB))
            plt.title('原始图像整体打码')
        if mosaic_mode in [1, 2]:
            plt.subplot(2, 3, 2)
            plt.imshow(cv2.cvtColor(keying_masked_image, cv2.COLOR_BGR2RGB))
            plt.title('需要打码的区域（未打码）')
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(mosaic_masked_image, cv2.COLOR_BGR2RGB))
        plt.title(f'（只有掩膜部分的）打码后的图像（mode {mosaic_mode}）')
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(keying_image, cv2.COLOR_BGR2RGB))
        plt.title('去掉（需要打码的）掩膜部分的图像')
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
        plt.title(f'打码的图像（打码模式：{mosaic_mode}，透明度：{alpha}）')
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        plt.title('分割后的打码图像')
        fig = plt.gcf()
        plt.show()

    # 显示原图像和拼接后的打码图像（没有框和分割区域）
    cv2.imshow("Original Image or Video", orig_img0)
    if draw_mosaic:
        cv2.imshow("Mosaic Image or Video", mosaic)

    if is_picture_type:
        return orig_img0, mosaic, annotated_frame, fig
    else:
        return orig_img0, mosaic, annotated_frame


def YOLOv8_Inference(model, file_paths, is_save=is_save0, picture_mosaic=True, picture_all=True,
                     video_mosaic=True, video_all=False):

    # file_paths: 所有需要推理的文件对应的路径（可以混合图片路径和视频（录像）路径）
    # is_save: 保存状态（见前面代码的说明）
    # picture_mosaic: 是否显示cv2窗口下的打码图像（没有框和分割区域）
    # picture_all: 是否显示figure窗口: 打码过程中各个阶段对应的图像（非视频）
    # video_mosaic: 是否显示cv2窗口下的打码视频（没有框和分割区域）
    # video_all: 是否显示figure窗口: 打码过程中各个阶段对应的图像(视频中的某一帧)

    # mosaic（打码）
    for index, file_path in enumerate(file_paths):
        # 提取文件名
        file_name = os.path.basename(file_path)
        # 是否为图片
        is_picture_type = file_name.lower().endswith(image_type)
        # 输出文件路径
        if file_path != '0':
            print(f"路径：{file_path}")

        # 路径为"0"时为摄像头
        if file_path == '0':
            print("摄像头：")
        else:
            print(f"文件名：{file_name}")

        # 文件类型
        if is_picture_type:
            print("文件类型：图片")
        elif file_path != '0':
            print("文件类型：视频")

        # 透明度
        print(f"掩膜部分的透明度：{alpha}")

        # 判断是否为图片
        if is_picture_type:

            # 对图片进行推理

            # 读取一张图片
            frame = cv2.imread(file_path)
            # 图片测试
            result = model(frame)
            # 判断推理结果是否没有物体，有就可能会显示原图、打码的图像、打码的推理结果、figure窗口中的打码过程；反之，只可能显示、保存原图。
            if len(result[0].boxes.cls) > 0:
                # 打码过程: 返回原图、打码的图像、打码的推理结果、figure窗口
                orig_img0, mosaic, annotated_frame, fig = prediction_mosaic(result, frame,
                                                                            picture_mosaic, picture_all, is_picture_type)
                # 显示cv2窗口中打码的推理结果
                cv2.namedWindow("YOLOv8 Inference (picture)", cv2.WINDOW_NORMAL)
                cv2.imshow("YOLOv8 Inference (picture)", annotated_frame)
                # 按任意键可关闭cv2窗口
                cv2.waitKey()

                # 保存图片
                if is_save['save original image']:
                    if not os.path.exists(os.path.join(img_dir_begin, file_name.split(".")[0])):
                        os.mkdir(os.path.join(img_dir_begin, file_name.split(".")[0]))
                    img_dir = os.path.join(img_dir_begin, file_name.split(".")[0])
                    img_name = 'Original image of ' + file_name.split(".")[0] + ".jpg"
                    cv2.imwrite(os.path.join(img_dir, img_name), orig_img0)
                if is_save['save mosaic image']:
                    if not os.path.exists(os.path.join(img_dir_begin, file_name.split(".")[0])):
                        os.mkdir(os.path.join(img_dir_begin, file_name.split(".")[0]))
                    img_dir = os.path.join(img_dir_begin, file_name.split(".")[0])
                    img_name = f'Mosaic image (mode {mosaic_mode} alpha {alpha})' + ".jpg"
                    cv2.imwrite(os.path.join(img_dir, img_name), mosaic)
                if is_save['save image of YOLOv8 inference']:
                    if not os.path.exists(os.path.join(img_dir_begin, file_name.split(".")[0])):
                        os.mkdir(os.path.join(img_dir_begin, file_name.split(".")[0]))
                    img_dir = os.path.join(img_dir_begin, file_name.split(".")[0])
                    img_name = f'Inference (mode {mosaic_mode} alpha {alpha})' + ".jpg"
                    cv2.imwrite(os.path.join(img_dir, img_name), annotated_frame)
                if is_save['save figure']:
                    if not os.path.exists(os.path.join(img_dir_begin, file_name.split(".")[0])):
                        os.mkdir(os.path.join(img_dir_begin, file_name.split(".")[0]))
                    img_dir = os.path.join(img_dir_begin, file_name.split(".")[0])
                    img_name = f'Figure (mode {mosaic_mode} alpha {alpha})' + ".jpg"
                    fig.savefig(os.path.join(img_dir, img_name))
            else:
                # 显示cv2窗口中的原图像
                cv2.imshow("Original Image or Video", frame)
                cv2.waitKey()
                if is_save['save original image']:
                    if not os.path.exists(os.path.join(img_dir_begin, file_name.split(".")[0])):
                        os.mkdir(os.path.join(img_dir_begin, file_name.split(".")[0]))
                    img_dir = os.path.join(img_dir_begin, file_name.split(".")[0])
                    img_name = 'Original image of ' + file_name.split(".")[0] + ".jpg"
                    cv2.imwrite(os.path.join(img_dir, img_name), frame)

        else:

            # 对视频（录像）进行推理

            if file_path == '0':
                # 捕捉摄像头（录像）
                cap = cv2.VideoCapture(0)
            else:
                # 捕捉视频
                cap = cv2.VideoCapture(file_path)

            # 视频尺寸
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # 视频帧率
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # 保存视频（录像）
            dir_name = 'webcam' if file_name == '0' else file_name.split(".")[0]
            if is_save['save original video']:
                if not os.path.exists(os.path.join(video_dir_begin, dir_name)):
                    os.mkdir(os.path.join(video_dir_begin, dir_name))
                video_dir = os.path.join(video_dir_begin, dir_name)
                video_name = 'Original video of ' + dir_name + ".mp4"
                out1 = cv2.VideoWriter(os.path.join(video_dir, video_name), cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, size)
            if is_save['save mosaic video']:
                if not os.path.exists(os.path.join(video_dir_begin, dir_name)):
                    os.mkdir(os.path.join(video_dir_begin, dir_name))
                video_dir = os.path.join(video_dir_begin, dir_name)
                video_name = f'Mosaic video (mode {mosaic_mode} alpha {alpha})' + ".mp4"
                out2 = cv2.VideoWriter(os.path.join(video_dir, video_name), cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, size)
            if is_save['save video of YOLOv8 inference']:
                if not os.path.exists(os.path.join(video_dir_begin, dir_name)):
                    os.mkdir(os.path.join(video_dir_begin, dir_name))
                video_dir = os.path.join(video_dir_begin, dir_name)
                video_name = f'Inference (mode {mosaic_mode} alpha {alpha})' + ".mp4"
                out3 = cv2.VideoWriter(os.path.join(video_dir, video_name), cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, size)

            # 检查当前的cv2.VideoCapture是否已经打开
            while cap.isOpened():
                # 获取视频（录像）中每一帧的图像
                res, frame = cap.read()
                # 如果读取成功
                if res:
                    # 正向推理
                    result = model(frame)
                    # 判断推理结果是否没有物体，有就可能会显示原视频、打码的视频、打码的推理结果、figure窗口；反之，只显示原视频
                    # （打码的视频和推理结果和原视频一样，对应的cv2窗口仍在）。如果不加这个判断，可能会在没有物体时报错。
                    if len(result[0].boxes.cls) > 0:
                        # 判断是否保存视频：只有当没有同时保存原视频、打码的视频、打码的推理结果时，not_save_video=True，即没有保存视频
                        not_save_video = (not is_save['save original video'] and not is_save['save mosaic video'] and
                                          not is_save['save video of YOLOv8 inference'])
                        # 打码过程: 输出原视频、打码的视频、打码的推理结果
                        orig_img0, mosaic, annotated_frame = prediction_mosaic(result, frame, video_mosaic,
                                                                                  video_all, is_picture_type,
                                                                                  not_save_video=not_save_video)
                        # 显示cv2窗口中打码的推理结果
                        cv2.namedWindow("YOLOv8 Inference (video)", cv2.WINDOW_NORMAL)
                        cv2.imshow("YOLOv8 Inference (video)", annotated_frame)

                        # 保存视频（录像)
                        if is_save['save original video']:
                            out1.write(orig_img0)
                        if is_save['save mosaic video']:
                            out2.write(mosaic)
                        if is_save['save video of YOLOv8 inference']:
                            out3.write(annotated_frame)
                    else:
                        # 只显示原视频
                        cv2.imshow("Original Image or Video", frame)
                        cv2.imshow("Mosaic Image or Video", frame)
                        cv2.imshow("YOLOv8 Inference (video)", frame)
                        # 保存视频（录像)
                        if is_save['save original video']:
                            out1.write(frame)
                        if is_save['save mosaic video']:
                            out2.write(frame)
                        if is_save['save video of YOLOv8 inference']:
                            out3.write(frame)

                    # 按a~z或者空格可以退出cv2窗口（在输入法为中文时按字母无效）
                    if (cv2.waitKey(1) & 0xFF) in [ord(" ")] + [ord("a") + i for i in range(0, 26)]:
                        break
                else:
                    break
            # 释放链接
            cap.release()
        # 销毁所有窗口
        cv2.destroyAllWindows()
        print('\n')


if __name__ == '__main__':

    # 模型权重
    # 自己做的数据集
    # model = YOLO('./runs/segment/CHIP_train2/weights/best.pt')
    model = YOLO('./runs/segment/coco128-seg_train5/weights/best.pt')

    # 所有需要推理的文件对应的路径
    # file_paths = ['./dataset/human_body/images/'+x for x in ['0001031.jpg', '0001674.jpg', '0001839.jpg',
    #              '0002085.jpg', '0002090.jpg', '0004130.jpg']]

    # file_paths = ['predict_files/video/b站/蔡徐坤 只因你太美.mp4']

    # file_paths = ['predict_files/picture/'+x for x in ['行人1.webp', '行人2.jpg', '行人3.jpg',
    #              '行人4.jpg', '行人5.webp', '行人6.jpg']]

    # file_paths = ['predict_files/picture/'+x for x in ['人群1.jpg', '人群2.webp', '人群3.webp']]

    # file_paths = ['./dataset/human_body/images/0001031.jpg']

    file_paths = ['predict_files/video/素材/人群和行人/【高清视频素材】香港城市行人交通时间流逝.mp4']

    # Predict（含打码）
    YOLOv8_Inference(model, file_paths, is_save=is_save)

