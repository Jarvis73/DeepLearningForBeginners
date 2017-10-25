# Suggestive Annotation

## Part 1 Data Processing

### 1. Get Segmentation from LIDC-IDRI

**标准：**  
* 不考虑只标注了一点的结节
* 只保留至少两个专家做了分割的结节
* 如果两个结节其中一个包含了另一个中心, 则视为同一个结节

### 2. Extract cubes from original medical images

* 图像重采样, 一个像素代表 0.7mm, 线性插值
* 提取目标区域, 大小为 64x64x64
