深度学习框架目标检测算法如何使用YOLOv8模型训练行人目标检测数据集权重模型 通过训练出的行人入侵检测数据集的权重 建立深度学习的行人入侵检测系统
@[toc]

数据集描述：
行人目标检测图像数据集yolo格式（图像.jpg+标签.txt）目标检测，训练YOLOv5、YOLOv8等![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/db7f6ddb6cd34545807212eb166d5676.jpeg)


已按7：2：1的比例划分好训练集（3056张）：验证集（873张）：测试集（436）

类别：1类：person
YAML文件![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/314ad6c76388422bbba7b59358c3791e.png)


使用行人目标检测数据集训练YOLOv5或YOLOv8模型，同学们。我们需要确保数据准备正确，并且按照适当的步骤进行模型训练、验证和测试。

以下是详细的指南，包括如何组织数据、编写YAML文件以及训练代码。

代码示例，仅供参考。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e384bf4d9670478caf1eb82cbbee6326.jpeg)

### 数据组织

数据已经按7:2:1的比例划分好，目录结构应该如下所示：

```
pedestrian_detection_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

每个`.jpg`图像文件都有一个对应的`.txt`标签文件，格式为YOLO（类别ID 中心x 中心y 宽度 高度），所有坐标均归一化。

### YAML配置文件

创建一个名为`data.yaml`的文件，内容如下：

```yaml
train: ./pedestrian_detection_dataset/images/train
val: ./pedestrian_detection_dataset/images/val
test: ./pedestrian_detection_dataset/images/test

nc: 1  # 类别数量
names: ['person']  # 类别名称
```

### 训练YOLOv8模型

#### 环境搭建

首先，确保你安装了必要的库。对于YOLOv8，你可以通过以下命令安装Ultralytics包：

```bash
pip install ultralytics
```

#### 开始训练

使用YOLOv8进行训练的Python脚本示例如下：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8s.pt')  # 或者选择其他预训练模型如 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'

# 训练模型
results = model.train(data='path/to/data.yaml',  # data.yaml 文件路径
                      imgsz=640,                  # 输入图片尺寸
                      epochs=100,                 # 训练轮数
                      batch=16,                   # 批次大小
                      name='pedestrian_detection', # 实验名称
                      device='0' if torch.cuda.is_available() else 'cpu')  # 使用GPU或CPU
```

如果你想直接从命令行运行训练命令，可以使用如下命令：

```bash
yolo task=detect mode=train model=yolov8s.pt data=path/to/data.yaml epochs=100 imgsz=640 batch=16 name=pedestrian_detection device=0
```

### 模型评估

训练完成后，可以在验证集上评估模型性能：

```bash
yolo task=detect mode=val model=runs/detect/pedestrian_detection/weights/best.pt data=path/to/data.yaml
```

这将输出模型的精度指标，例如 mAP@0.5, mAP@0.5:0.95 等。

### 推理与部署

#### 单图推理

```python
from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO('runs/detect/pedestrian_detection/weights/best.pt')
results = model('path/to/image.jpg')

for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])  # 转换颜色通道顺序
    im.show()  # 显示结果
```

#### 视频或摄像头实时检测

```python
def detect_video(video_path=0):  # 默认使用摄像头
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Pedestrian Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 q 键退出
            break
    cap.release()
    cv2.destroyAllWindows()

# 使用示例
detect_video()  # 使用摄像头
# 或者指定视频路径
# detect_video('path/to/video.mp4')
```

### 模型导出

为了便于在不同平台上部署，可以将模型导出为不同的格式：

#### 导出为 ONNX 格式

```bash
yolo export model=best.pt format=onnx
```

#### 导出为 TensorRT（NVIDIA 设备）

```bash
yolo export model=best.pt format=engine device=0
```

#### 导出为 OpenVINO（Intel 设备）

```bash
yolo export model=best.pt format=openvino
```

通过上述步骤，行人目标检测数据集训练YOLOv8模型，
