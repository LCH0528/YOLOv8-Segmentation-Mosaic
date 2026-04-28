# YOLOv8-Segmentation-Mosaic: Intelligent Blurring Tool Based on the YOLOv8 Semantic Segmentation Model

## 📋 Project Overview

This project is based on the YOLOv8 semantic segmentation model and can be used to blur (mask) segmentation regions (masks) of **specified classes** in the model’s output. The tool supports processing images and videos (including real-time camera streams) and offers three blurring modes: **pixel mosaic**, **monochrome mosaic**, and **multicolor mosaic**.

> - By setting the **category index** (e.g., `mosaic_index = [0,5,8]`), users can apply masking to all segmentation regions belonging to that category. The masked area exactly matches the boundaries of the segmentation mask predicted by the model. The accuracy of the masking directly depends on the quality of the underlying YOLOv8 segmentation model’s inference: the more accurately the model segments, the better the masked area aligns with the object’s outline.
> - This project originated as a small-scale research project during my graduate studies. It has undergone iterative development through three versions—`prediction_mosaic.py`, `prediction_mosaic_dlc.py`, and `prediction_mosaic_dlc_up.py`—with its functionality gradually refined. **For complete technical details, implementation principles, and the development process, please refer to the project's [three codes](code/YOLOv8_mosaic/ultralytics-main), [PPT folder](PPT和作图) and [comprehensive instructional videos](视频说明).**
> - The blurring code can be applied to images, videos, or live camera feeds. You can pass in a list of file paths (which may include a mix of these three file types). When the code runs, it will blur each file in the order of the file paths (for videos, it blurs individual frames and then reassembles them into a single video; the same applies to live camera feeds).

---

## 📁 Project Structure

<details open>
<summary><b>Core Files and Directories</b></summary>

The following is the main directory structure for this project. The core source code and data are located in the `📂 ultralytics-main/` folder.

```plaintext
YOLOv8-Segmentation-Mosaic/
├── 📂 code/                  # Source Code Directory
│   └──  📂 YOLOv8_mosaic/
│        ├── 📂 pycococreator-master/  # COCO Dataset Format Conversion Toolkit
│        └──  📂 ultralytics-main/   # Core project code and data
│             ├── 📂 dataset/   # Datasets
│             ├── 📂 predict_files/   # Sample images and videos for reasoning tests
│             ├── 📂 runs/   # Training and validation results (including weight files)
│             ├── 📄 prediction_mosaic.py        # Watermarking Program (v1.0 Basic Edition)
│             ├── 📄 prediction_mosaic_dlc.py    # Code-masking Program (v2.0 Enhanced Edition)
│             ├── 📄 prediction_mosaic_dlc_up.py # Code-masking Program (v3.0 Optimized Version)
│             ├── 📄 jpg2png.py       # Image Format Converter (JPG → PNG)
│             ├── 📄 json2txt.py      # Annotation Format Converter（Labelme JSON → YOLO TXT）
│             ├── 📄 json2txt_show.py    # Label Visualization Tool（Labelme JSON → YOLO TXT）
│             ├── 📄 mask2coco.py        # Annotation Format Converter（Mask → COCO JSON）
│             ├── 📄 spilit_dataset.py   # Dataset splitting script (training set/validation set)
│             └── 📄 results_plt.py      # Visualization of training results curves
├── 📂 PPT和作图/              # Project Presentation PPT
├── 📂 视频说明/               # Feature Overview and Demo Videos
├── 📂 demo/                 # Store all presentation materials (GIFs, MP4s, etc.)
├── 📄 README.md        # Project Description Document (English)
└── 📄 README_zh.md     # Project Description Document (Chinese)
```
</details>

---

## 📁 Evolution of Core Document Versions
This project includes three core version files that illustrate the gradual evolution of its features:

|         Version         |             File              |            Supports blurring mode             |                  Key Features                   | Rating |
|:-----------------------:|:-----------------------------:|:-----------------------------:|:---------------------------------------:|:------:|
| v1.0<br/> Basic Edition |    `prediction_mosaic.py`     |      Supports only pixel mosaic <br/> (Mode 0)       |     Basic blurring functionality supports only pixel mosaic; does not support transparency adjustment.      |  ★☆☆   |
|  v2.0 Enhanced Edition  |  `prediction_mosaic_dlc.py`   |  Supports pixel, monochrome, and multicolor mosaics <br/> (modes 0, 1, 2)  |      Supports three blurring modes with customizable colors; supports transparency adjustment.      |  ★★☆   |
| v3.0 Optimized Version  | `prediction_mosaic_dlc_up.py` |  Supports pixel, monochrome, and multicolor mosaics <br/> (modes 0, 1, 2)  |  Based on the v2.0 Enhanced Edition, color settings have been standardized, and the configuration at is now more streamlined. We recommend using this version.  |  ★★★   |

---

## ✨ Three Encoding Modes

### 🎨 Mode 0: Pixel Mosaic
- **Implementation Principle**: By downscaling and then upscaling using nearest-neighbor interpolation, a blocky mosaic effect is created.
- **Key Parameters**: `mosaic_size` (mosaic block size)

### 🎨 Pattern 1: Monochrome Mosaic
- **Implementation Principle**: All masking areas across all categories are filled with the same color
- **Color Settings**: Supports RGB, 8 basic colors, and default mask color

### 🎨 Pattern 2: Multicolored Mosaic
- **Implementation Principle**: Different categories of masked areas are filled with different colors
- **Color Settings**: Supports RGB, 8 basic colors, and default mask color

<br/>
<details open>
<summary><b>Schematic Diagram of Pixel Mosaic Generation</b></summary>
           
<br/>
           
![Schematic Diagram of Pixel Mosaic Generation](PPT和作图/2024.6.2/2024.6.2_21.png)
</details>

<br/>
<details open>
<summary><b>Renderings of pixel, monochrome, and multicolor mosaic effects</b></summary>
           
<br/>

![Renderings of 3 watermarking modes](PPT和作图/2024.6.17/作图/打码模块.png)
</details>

---

## 🚀 Quick Start

### 1. Environment Setup

> Simply follow the installation instructions for the YOLOv8 environment.
> <br/>Reference video: https://www.bilibili.com/video/BV13V4y1S7MK/?spm_id_from=333.1391.0.0&vd_source=3e34b6ee0b8dc3763064b64eac827b5a

- **https://github.com/ultralytics/ultralytics && unzip ultralytics-main.zip && cd ultralytics-main**
- **Install dependencies:**
```bash 
pip install -e
```

### 2. Configuration File (Settings)

#### 🔧 Basic Parameters (Common to All Versions)
Before discussing the differences between the versions, the following parameters are core settings common to all three versions; they determine the target categories for text masking, saving settings, and more.

> **The `file_paths` field can contain the paths to all files required for inference, including images, videos, and camera feeds (`‘0’`).**

```python
nc = 19     # Number of target categories in the dataset

# Category numbers requiring redaction (starting from 0)
# mosaic_index = [1, 9, 12]  # CHIP and human_body
mosaic_index = [0]   # COCO

# Partial path to the image
img_dir_begin = './predict_files/predict/human_body-seg/picture'

# Partial path for saving videos (recordings)
video_dir_begin = './predict_files/predict/human_body-seg/video'

# Save State
is_save = {'save original image': False,
           'save mosaic image': False,
           'save image of YOLOv8 inference': False,
           'save figure': False,
           'save original video': False,
           'save mosaic video': False,
           'save video of YOLOv8 inference': False}

# Save Options:
# ‘save original image’: Save the original image
# ‘save mosaic image’: Save the pixelated image
# ‘save image of YOLOv8 inference’: Save the YOLOv8 segmentation results (including pixelation)
# ‘save figure’: Save a figure window (image format) containing all images from the pixelation process
# ‘save original video’: Save the original video
# ‘save mosaic video’: Save the pixelated video
# ‘save video of YOLOv8 inference’: Save the YOLOv8 segmentation results for the video (including pixelation)

...

if __name__ == '__main__':
    # Model Weights
    model = YOLO('./runs/segment/coco128-seg_train5/weights/best.pt')
    
    # The paths corresponding to all files that require inference
    file_paths = ['predict_files/picture/游泳4.jpg', 'predict_files/video/老实巴蕉.mp4', 'predict_files/picture/游泳男.webp',
                  '0', 'predict_files/picture/其他2.jpg', 'predict_files/picture/其他3.jpeg']
```

<br/>

#### 🔧 Version-specific settings
Below are the specific differences between the three versions in terms of pixelation settings, color schemes, and other aspects.

<details open>
<summary><b>v1.0 Basic Edition (prediction_mosaic.py) Parameter Settings</b></summary>

<br/>

```python
# Watermark size (size of the image before resizing)
mosaic_size = 64
```

</details>

<details open>
<summary><b>v2.0 Enhanced Edition (prediction_mosaic_dlc.py) Parameter Settings</b></summary>

<br/>

```python
# Pattern Configuration
mosaic_mode = 2  # 0=Pixel, 1=Monochrome, 2=Multicolor
alpha = 1  # Transparency

# Mode 0 (Pixel) Settings
mosaic_size = 64    # Watermark size (size of the image before resizing)

# Mode 1 (Monochrome) Settings
self_color = False # Use a custom single color: True corresponds to self_color, False corresponds to the default color
single_color_select = "g"  # Use preset colors
# ---OR---
self_color = True
self_mosaic_color = "#FF1493"  # Custom Colors

# Mode 2 (Multicolor) Settings
self_colors = True # Whether to use multiple colors from custom categories: True corresponds to self_colors, False corresponds to mask colors
from_single_color = True # Do the custom colors I set all come from the basic colors in (monochrome marking mode)?
# Custom colors for different categories (using custom RGB values), where each row corresponds to a category that needs to be masked.
# For each category, you can enter an RGB value as a list, such as [255, 255, 255], or in RGB hexadecimal string format (e.g., “#??????”),
# such as “#FF0000” (do not omit the #; ? represents 0, 1, 2, ..., 9, A, B, C, D, E, F).
self_mosaic_colors = [[128, 128, 128],
                      "#FF1493",
                      [0, 255, 255]]
# ---OR---
# A list composed of primary colors (where each element is a letter corresponding to a color) used to map to the corresponding RGB values
single_color_list = ["g", "r", "c"]  # Each category corresponds to a color
```

</details>

<details open>
<summary><b>v3.0 Optimized Version (prediction_mosaic_dlc_up.py) Parameter Settings</b></summary>
           
<br/>
           
> **Key Improvements**: The v3.0 Optimized Version consolidates the Mode 1 and Mode 2 color settings—which were separate in the v2.0 Enhanced Edition—into a single `self_mosaic_colors` parameter. It automatically determines whether to use a single-color or multi-color mode based on the length of the list, greatly simplifying configuration.

<br/>

```python
# Mode Configuration
mosaic_mode = 1  # 0 = pixel, 1 = monochrome/Multicolor
alpha = 1  # Transparency

# Mode 0 Exclusive Settings
mosaic_size = 64  # Mosaic Block Size

# Unified Color Settings (Mode 1)
# self_mosaic_colors: A custom set of colors for different redaction categories, where each row corresponds to a category requiring redaction.
# Each row in self_mosaic_colors can take one of the following forms; self_mosaic_colors must be a two-dimensional list ([[?],[?],[?]]), and each row’s list can also be a string:
# (1) A row in `self_mosaic_colors` contains a list of three numbers representing RGB values (these can be specific integer grayscale values between 0 and 255; they can also be normalized grayscale values within the range of 0 to 1,
# which subsequent code can convert to specific values). For example: a row [255, 0.7, 0].
# (2) A row in `self_mosaic_colors` contains an RGB hex string format (`“#??????”`), such as `“#FF0000”` (do not omit the `#`; `?` represents 0, 1, 2, ..., 9, A, B, C, D, E, F).
# (3) A row in `self_mosaic_colors` contains strings corresponding to the default 8 basic colors (keys in the `single_color` dictionary), such as “g”, ‘r’, “c”, which can be used to map to the corresponding RGB values.
# (4) `self_mosaic_colors=[[]]` (empty list): Indicates that colors from the default mosaic color table will be used for encoding.
# The following are examples of `self_mosaic_colors` settings (multiple formats can be mixed):
# self_mosaic_colors = [[128, 0.5, 0],
#                       "#FF1493",
#                       "g"]

# ---Example---
# Configuration Method 1: Monochrome Mode
self_mosaic_colors = ["#FF0000"]  # All categories in red

# Configuration Method 2: Multi-color Mode
self_mosaic_colors =  [[128, 0.5, 0],
                      "#FF1493",
                      "g"]  # Use different colors for different categories (you can mix and match in various ways)

# Configuration Method 3: Using the Default Color Palette
self_mosaic_colors = [[]]  # Automatically assigns 20 different colors
```

</details>

### 3. Run the program

> The following example uses the optimized version, `prediction_mosaic_dlc_up.py`

```bash
python prediction_mosaic_dlc_up.py
```

### 4. Human-Computer Interaction
#### Window Control
- **Figure Window:** A window displayed by the program (e.g., showing a step-by-step diagram of the blurring process). Click the close button (×) in the upper-right corner of the window to close it; the program will then continue with the subsequent steps.
- **CV2 Window (Image):** When processing images, multiple CV2 windows will appear. Press any key on the keyboard to close all open CV2 windows simultaneously (such as the original image, the blurred image without inference results, and the blurred image with inference results). The program will then continue with the subsequent process.
- **CV2 Window (Video/Camera):** When processing video or a live camera feed, the video will play continuously. Press the spacebar or any letter key (a-z) to immediately stop playback and close all CV2 windows simultaneously; the program will then continue with the subsequent process.

#### Save Output
- **Video (Camera) Recording Duration Notes:** When processing video or camera footage, the recording duration of the output file ends at the moment the user actively presses the exit key (spacebar or any letter). If no key is pressed by the time the video finishes playing, the entire video will be saved.
- **Output File Control:** By modifying the Boolean values in the `is_save` configuration dictionary, you can precisely control the types of output files to be saved (e.g., original files, pure redaction results, or redaction results with segmented output).
- **Configuration Example:** Assuming the following command-line arguments are set:
  - Input file paths: `file_paths = [‘predict_files/picture/image1.jpg’, ‘predict_files/video/video1.mp4’, ‘0’]` (where ‘0’ represents the camera)
  - Partial path for saving images: `img_dir_begin = ‘./predict_files/predict/CIHP/picture’`
  - Partial path for saving videos (`video_dir_begin = ‘./predict_files/predict/CIHP/video’`)
  - Censoring mode (`mosaic_mode = 0`; pixel mosaic as an example), transparency (`alpha = 1`)
  - Set all values in the `is_save` configuration dictionary to True

<details open>
<summary><b>After running the program, the following file structure will be generated</b></summary>

<br/>

```plaintext
📂 predict_files/             # Input file and output directory (input files can also be specified from other paths)
├── 📂 picture/               # Input image
├── 📂 video/                 # Source video
└── 📂 predict/               # Inference Results
    └── 📂 CIHP/              # Inference results using weights from the CIHP dataset
        └── 📂 picture/       # The result of blurring the image
            └── 📂 图片1/       # Blurred version of图片1.jpg
                ├── 📄 Original image of 图片1.jpg              # 图片1.jpg
                ├── 📄 Mosaic image (mode 0 alpha 1).jpg       # Blurred version of 图片1.jpg (without showing the segmentation results)
                ├── 📄 Inference (mode 0 alpha 1).jpg          # Blurred version of 图片1.jpg (Display the segmentation results simultaneously)
                └── 📄 Figure (mode 0 alpha 1).jpg             # Process Flowchart (Figure Window)     
        └── 📂 video/         # Blurring results for input video (Webcam)
            └── 📂 视频1/       # Blurred version of 视频1.mp4
                ├── 📄 Original video of 视频1.mp4              # 视频1.mp4
                ├── 📄 Mosaic video (mode 0 alpha 1).mp4       # Blurred version of 视频1.mp4 (without showing the segmentation results)
                └── 📄 Inference (mode 0 alpha 1).mp4          # Blurred version of 视频1.mp4 (Display the segmentation results simultaneously)
            └── 📂 webcam/     # Real-time blurring results from the webcam
                ├── 📄 Original video of webcam.mp4            # Webcam input
                ├── 📄 Mosaic video (mode 0 alpha 1).mp4       # Blurred results from the Webcam input (without showing the segmentation results)
                └── 📄 Inference (mode 0 alpha 1).mp4          # Blurred results from the Webcam input (Display the segmentation results simultaneously)
```
</details>

---

<br/>

## 📸 Sample of the blurring effect

<details open>
<summary style="font-size: 1.17em;"><b>1. Image blurring</b></summary>

<!-- [![Input image](code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Original image of 其他.jpg)](code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Original image of 其他.jpg) -->

<div align="center">
<h3>Input image</h3>
  <a href="code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Original%20image%20of%20其他.jpg" target="_blank">
    <img src="code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Original image of 其他.jpg" alt="Input image" style="max-width: 100%; height: auto;">
  </a>
</div>

<br/>

<h3 align="center">Blurring effect</h3>

| Should the partitioning results be displayed at the same time? |                                                                                                                                                                              Pixel mosaic                                                                                                                                                                               |                                                                                                                                                                     Solid-color mosaic                                                                                                                                                                      |                                                                                                                                                                       Multicolored mosaic                                                                                                                                                                        |
|:--------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                  **No<br>(Placeholder text)**                  | <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Mosaic%20image%20(mode%200%20alpha%201).jpg" target="_blank"><img src="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Mosaic%20image%20(mode%200%20alpha%201).jpg" alt="Pixel mosaic" style="width: 100%; max-width: 240px; border-radius: 4px;"></a> | <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Mosaic%20image%20(mode%201%20alpha%201).jpg" target="_blank"><img src="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Mosaic%20image%20(mode%201%20alpha%201).jpg" alt="Solid-color mosaic" style="width: 100%; max-width: 240px; border-radius: 4px;"></a> | <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Mosaic%20image%20(mode%202%20alpha%201).jpg" target="_blank"><img src="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Mosaic%20image%20(mode%202%20alpha%201).jpg" alt="Multicolored mosaic" style="width: 100%; max-width: 240px; border-radius: 4px;"></a> |
|          **Yes<br>(Including split-screen display)**           |      <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Inference%20(mode%200%20alpha%201).jpg" target="_blank"><img src="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Inference%20(mode%200%20alpha%201).jpg" alt="Pixel mosaic" style="width: 100%; max-width: 240px; border-radius: 4px;"></a>      |   <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Inference%20(mode%201%20alpha%201).jpg" target="_blank"><img src="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Inference%20(mode%201%20alpha%201).jpg" alt="Solid-color mosaic" style="width: 100%; max-width: 240px; border-radius: 4px;"></a>    |      <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Inference%20(mode%202%20alpha%201).jpg" target="_blank"><img src="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/picture/其他/Inference%20(mode%202%20alpha%201).jpg" alt="Multicolored mosaic" style="width: 100%; max-width: 240px; border-radius: 4px;"></a>      |

<p align="center">Click any image to view the original</p>

</details>

<br/>

<details open>
<summary style="font-size: 1.17em;"><b>2. Video blurring</b></summary>

<div align="center">
<h3>Input video</h3>
  <a href="code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/video/香蕉君鱿鱼游戏一命速通（转载）/Original%20video%20of%20香蕉君鱿鱼游戏一命速通（转载）.mp4" target="_blank">
    <img src="demo/香蕉君鱿鱼游戏一命速通%EF%BC%88转载%EF%BC%89/gif/screentogif/Online-Convert/Original%20video%20of%20香蕉君鱿鱼游戏一命速通%EF%BC%88转载%EF%BC%89.gif" alt="Input video" style="max-width: 100%; height: auto;">
  </a>
</div>

<br/>

<h3 align="center">Blurring effect</h3>

| Should the partitioning results be displayed at the same time? |                                                                                                                                                                        Pixel mosaic                                                                                                                                                                         |                                                                                                                                                                    Solid-color mosaic                                                                                                                                                                    |                                                                                                                                                                    Multicolored mosaic                                                                                                                                                                    |
|:--------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                  **No<br>(Placeholder text)**                  |     <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/video/香蕉君鱿鱼游戏一命速通（转载）/Mosaic%20video%20(mode%200%20alpha%201).mp4" target="_blank"><img src="./demo/香蕉君鱿鱼游戏一命速通（转载）/gif/screentogif/Online-Convert/Mosaic video (mode 0 alpha 1).gif" alt="Pixel mosaic" style="width: 100%; max-width: 200px; border-radius: 4px;"></a>      | <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/video/香蕉君鱿鱼游戏一命速通（转载）/Mosaic%20video%20(mode%201%20alpha%201).mp4" target="_blank"><img src="./demo/香蕉君鱿鱼游戏一命速通（转载）/gif/screentogif/Online-Convert/Mosaic video (mode 1 alpha 1).gif" alt="Solid-color mosaic" style="width: 100%; max-width: 200px; border-radius: 4px;"></a> | <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/video/香蕉君鱿鱼游戏一命速通（转载）/Mosaic%20video%20(mode%202%20alpha%201).mp4" target="_blank"><img src="./demo/香蕉君鱿鱼游戏一命速通（转载）/gif/screentogif/Online-Convert/Mosaic video (mode 2 alpha 1).gif" alt="Multicolored mosaic" style="width: 100%; max-width: 200px; border-radius: 4px;"></a> |
|          **Yes<br>(Including split-screen display)**           |         <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/video/香蕉君鱿鱼游戏一命速通（转载）/Inference%20(mode%200%20alpha%201).mp4" target="_blank"><img src="./demo/香蕉君鱿鱼游戏一命速通（转载）/gif/screentogif/Online-Convert/Inference (mode 0 alpha 1).gif" alt="Pixel mosaic" style="width: 100%; max-width: 200px; border-radius: 4px;"></a>          |     <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/video/香蕉君鱿鱼游戏一命速通（转载）/Inference%20(mode%201%20alpha%201).mp4" target="_blank"><img src="./demo/香蕉君鱿鱼游戏一命速通（转载）/gif/screentogif/Online-Convert/Inference (mode 1 alpha 1).gif" alt="Solid-color mosaic" style="width: 100%; max-width: 200px; border-radius: 4px;"></a>     |              <a href="./code/YOLOv8_mosaic/ultralytics-main/predict_files/predict/CIHP/video/香蕉君鱿鱼游戏一命速通（转载）/Inference%20(mode%202%20alpha%201).mp4" target="_blank"><img src="./demo/香蕉君鱿鱼游戏一命速通（转载）/gif/screentogif/Online-Convert/Inference (mode 2 alpha 1).gif" alt="Multicolored mosaic" style="width: 100%; max-width: 200px; border-radius: 4px;"></a>               |

<p align="center">Click any GIF to watch the original video</p>

</details>

---

<br/>

## 🗃️ Dataset

> The training results for the dataset used in this project are actually quite mediocre; we recommend that users find a more suitable dataset for separate training, validation, and inference.
> <br/>For a detailed description of the dataset used in this project, please refer to the PowerPoint presentation: [2024.6.2.pptx](PPT和作图/2024.6.2/2024.6.2.pptx), [2024.6.2（续）.pptx](PPT和作图/2024.6.2/2024.6.2（续）.pptx) and [2024.6.17（对不同数据集的介绍、训练和推理）.pptx](PPT和作图/2024.6.17/2024.6.17（对不同数据集的介绍、训练和推理）.pptx); as well as relevant links (such as CSDN, the official dataset website, etc.).

<details open>
<summary><b>Related Links</b></summary>

<br/>

**1. Introduction to the coco128-seg Dataset**

https://blog.csdn.net/babbycool/article/details/123115899

https://blog.csdn.net/weixin_51031772/article/details/135679015

https://blog.csdn.net/oYeZhou/article/details/112008054

https://blog.csdn.net/XDH19910113/article/details/125299757

**2. An Introduction to Datasets Such as CHIP**

https://blog.csdn.net/wxf19940618/article/details/83661891

https://blog.csdn.net/qq_41994006/article/details/126191667

https://blog.csdn.net/weixin_41809530/article/details/120237242

https://blog.csdn.net/hyk_1996/article/details/91974621

> Note: Datasets such as **CHIP** are segmentation datasets that include human body parts (or clothed torsos and exposed areas).

**3. Download links for datasets such as CHIP**

https://sysu-hcp.net/lip/overview.php

> If the download link is broken, you can try my Baidu Netdisk (which includes CHIP, LIP, MHP, BDD100K, CamVid, and more):
> <br/>Files shared via cloud storage: YOLOv8 mask code and dataset download Link: https://pan.baidu.com/s/1mWstrlA_aAaoivoZBXxg6g?pwd=99y4 
> Passcode: 99y4—Shared by a Baidu Netdisk Super Member v8

</details>

---

## 🎥 Full explanatory video

The full video tutorial for this project is available below:
- [prediction_mosaic.mp4](视频说明/prediction_mosaic.mp4)：`prediction_mosaic.py`（v1.0 Basic Edition）
- [prediction_mosaic_dlc和prediction_mosaic_dlc_up.mp4](视频说明/prediction_mosaic_dlc和prediction_mosaic_dlc_up.mp4)：`prediction_mosaic_dlc.py`（v2.0 Enhanced Edition）和`prediction_mosaic_dlc_up.py`（v3.0 Optimized Version）

> Since the full tutorial video is larger than 10 MB, it cannot be displayed directly on GitHub. You can access it using the following methods:
> <br/>**1. Download from GitHub**
> <br/>**2. Contact the author**
> <br/><b>QQ Email: </b>3524345723@qq.com

--- 

## ❓ Other questions

**Q1: Before adding the prediction box and mask colors, the blurring effect is acceptable; however, once they are added, the blurring color you set may blend with the mask color in the original image (resulting in a color that no longer matches the intended blurring color, although the blurring effect remains).<br/>**
**Q2: The actual range of instances may differ slightly from the mask.<br/>**
**A1&A2: Please refer to [2024.6.17（对指定多个类别打码）.pptx](PPT和作图/2024.6.17/2024.6.17（对指定多个类别打码）.pptx)and the links below: <br/>**
https://blog.csdn.net/weixin_46566149/article/details/136367700
> Remember to modify the [**plotting.py**](code/YOLOv8_mosaic/ultralytics-main/ultralytics/utils/plotting.py) file (it has already been modified in the uploaded project).
