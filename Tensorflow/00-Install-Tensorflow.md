# 安装 Tensorflow

Tensorflow 是一个使用 C/C++ 编写深度学习框架, 由 Google 开发, 目前有多种编程语言的软件包

* Python (主要支持)
* C++
* JavaScript
* Java
* Go
* Swift

同时也有多种语言的绑定(第三方支持)

* C#
* Haskell
* Julia
* Ruby
* Rust
* Scala

## 1. 安装 Python 版本的 Tensorflow

### 1.1 使用 pip 安装

Tensorflow 1.13.1 的 Python 版本有官方预打包好的库, 可以直接通过 pip 安装:

```bash
# 安装CPU版本
pip install tensorflow==1.13.1

# 安装GPU版本
pip install tensorflow-gpu==1.13.1
```

目前已经有 Tensorflow 2.0 预览版, 由于用法和 1.x 有较大差别, 暂不使用. 安装方法如下:

```bash
pip install tensorflow==2.0.0-alpha0 
```

### 1.2 使用 conda 安装

```bash
# 安装CPU版本
conda install tensorflow==1.13.1

# 安装GPU版本
conda install tensorflow-gpu==1.13.1
```

## 2. 启动测试

进入命令行, 键入 `python` 进入交互模式, 执行以下命令:

```python
import tensorflow as tf
print(tf.VERSION)

# 得到以下结果则说明安装正确:
'1.13.1'
```

基于静态图的基础计算:

```python
a = tf.constant(1)
b = tf.constant(2)
c = a + b
with tf.Session() as sess:
    c_val = sess.run(c)
print(c_val)

# 结果
3
```

打印 Hello World:

```python
hello = tf.constant("Hello World!")
with tf.Session() as sess:
    hello_val = sess.run(hello)
print(hello_val.decode())

# 结果
Hello World!
```

