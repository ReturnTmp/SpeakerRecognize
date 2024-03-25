League of Legends Speaker Recognition System

本仓库来自：https://github.com/jcfszxc/SpeakerRecognize

> 请务必先看 **最终解决** 章节

## 快速开始

1. Install CUDA
2. Install Anaconda3
3. Install dependencies
```python
conda create -n Deep_Speaker python=3.6
conda activate Deep_Speaker
conda install tensorflow-gpu=2.1.0 keras=2.3.1
conda install -c conda-forge pandas librosa
conda install -c conda-forge pyaudio
conda install -c bricew python_speech_features
```
4. Download the data and run training:
https://pan.baidu.com/s/1oI0Q8La1kNQr2V27BTswHA jcfs 

put data in './datasets/'

```python
python train.py
```

## Usage

pretrained model: https://pan.baidu.com/s/14RABQEDWwUrJ9CKOL7nStA jcfs
```python
python SpeakerRecog.pyw
```

上面网盘得到的预训练文件为 .h5 文件，一般情况下存放位置为 `~/.keras/models`，但本项目应该放在项目根目录 `checkpoints` 文件夹下 

如果卡在 Adding visible gpu devices: 0

设置环境变量 CUDA_CACHE_MAXSIZE=4294967296 即可

如果 numba 模块出现问题，卸载原有模块，执行 `pip install numba==0.48.0`

numpy 同理，`pip install numpy==1.17.0`

长时间卡在 successfully opened dynamic library libcudnn .so.7

官方 issue：https://github.com/tensorflow/tensor2tensor/issues/1643

大概率是CUDA、cuDNN、tensorflow的版本匹配问题

使用 nvcc -V 查看 cuda 版本，查看是否匹配

我自己的就是 cuda 11.7 ，但是 tensorflow 为 2.1，所以重新安装高版本

`conda install tensorflow-gpu=2.5`

但是注意，tensorflow 和 keras 版本是严格对应的（详情看：https://zhuanlan.zhihu.com/p/465947475 ）

所以需要 `conda install keras=2.5.0`

---
上面最终实测失败

## 最终解决

如果只是想要复现效果，就给上面的预训练模型下载之后，放在根目录 `checkpoints` 文件夹下 

然后对于 cuda 11.7 相近版本，直接按照下面步骤

### 环境

```bash
# 创建环境
conda create -n Deep_Speaker39 python=3.9
conda activate Deep_Speaker39
# 安装依赖
pip install tensorflow-gpu==2.10 keras -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install pandas librosa pyaudio python_speech_features pillow matplotlib -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```

### 执行

```python
python SpeakerRecog.pyw
```


 