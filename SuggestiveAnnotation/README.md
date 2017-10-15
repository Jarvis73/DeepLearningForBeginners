# Suggestive Annotation

## Part 1 Data Processing

### 1. Get Segmentation from LIDC-IDRI

**标准：**  
* 不考虑只标注了一点的结节
* 只保留至少两个专家做了分割的结节
* 如果两个结节其中一个包含了另一个中心, 则视为同一个结节

