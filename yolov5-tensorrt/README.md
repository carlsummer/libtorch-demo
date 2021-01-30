### tensorrt下载

> https://developer.nvidia.com/nvidia-tensorrt-7x-download
>
> https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/7.0/7.0.0.11/tars/TensorRT-7.0.0.11.CentOS-7.6.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz?rO5XVANjOZyyyHdIGV8uIKy0VqaeKj9Vec1hU47pyzSg57wL1irBZnGNK4rA96XxUw7LFSbkdUMy7pv8GQgJth2hImOmSCi49h5VahQ3p6b4mTRMxvrS4S-5NPAJFhN9xxgZjnjJjwqSs3ciFTJN2psxX3UvSkb7iSV4Q_02fXZNnrR_KICMrOIwp11nZ8RmFvCIDdQVb-D6wAb9Jd_weDwls5A0DjEcEnG0gshWecmVR1zj819ydGmORA
>
> CuDNN 7.6.5 CUDA Runtime 10.2 TorchVision: 0.7.0 torch 1.6.0

### 准备虚拟环境

> conda create -n py37_cu102_cunn7.6.5 --offline python=3.7.5
>
> conda activate py37_cu102_cunn7.6.5
>
> conda remove -n py37_cu102_cunn7.6.5 --all

### 安装tensorrt

> CUDA_HOME=/usr/local/cuda-10.2 
>
> cd TensorRT-7.0.0.11/python
> pip install tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl
>
> cd TensorRT-7.0.0.11/graphsurgeon
> pip install graphsurgeon-0.4.5-py2.py3-none-any.whl

```bash
export CUDA_HOME="/usr/local/cuda-10.2"
export TENSORRT_HOME="/home/deploy/software/TensorRT-7.0.0.11/"
export LD_LIBRARY_PATH="$TENSORRT_HOME/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CMAKE_HOME=/home/deploy/software/cmake-3.14.5-Linux-x86_64
export PATH="/home/deploy/anaconda3/bin:$PATH:$CUDA_HOME/bin:$CMAKE_HOME/bin"
. /home/deploy/anaconda3/etc/profile.d/conda.sh

```

### 验证是否安装成功

```shell
 cd software/TensorRT-7.0.0.11/samples/python/network_api_pytorch_mnist
 # 安装pip
 https://pypi.org/project/pip/19.1.1/#files
 cd pip-19.1.1
 python setup.py install
 # 继续测试
 pip install torch==1.6.0
 cd software/TensorRT-7.0.0.11/samples/python/network_api_pytorch_mnist
 pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
python sample.py
```

### yolov5 转tensorrt

```shell
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov5.git
// download its weights 'yolov5s.pt'
// copy tensorrtx/yolov5/gen_wts.py into ultralytics/yolov5
// ensure the file name is yolov5s.pt and yolov5s.wts in gen_wts.py
// go to ultralytics/yolov5
python gen_wts.py
// a file 'yolov5s.wts' will be generated.
```

```shell
// put yolov5s.wts into tensorrtx/yolov5
// go to tensorrtx/yolov5
// ensure the macro NET in yolov5.cpp is s
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
export CC=/usr/local/bin/gcc
export CXX=/usr/local/bin/g++
cmake3 ..
make
sudo ./yolov5 -s             // serialize model to plan file i.e. 'yolov5s.engine'
sudo ./yolov5 -d  ../samples // deserialize plan file and run inference, the images in samples will be processed.
```

```shell
// install python-tensorrt, pycuda, etc.
// ensure the yolov5s.engine and libmyplugins.so have been built
python yolov5_trt.py
```

### linux 中安装opencv

```shell
sudo yum install epel-release git gcc gcc-c++ cmake3 qt5-qtbase-devel \
    python python-devel python-pip cmake python-devel python34-numpy \
    gtk2-devel libpng-devel jasper-devel openexr-devel libwebp-devel \
    libjpeg-turbo-devel libtiff-devel libdc1394-devel tbb-devel numpy \
    eigen3-devel gstreamer-plugins-base-devel freeglut-devel mesa-libGL \
    mesa-libGL-devel boost boost-thread boost-devel libv4l-devel

https://github.com/opencv/opencv/releases

git clone https://github.com/opencv/opencv_contrib.git
tar -zxvf opencv_contrib-4.5.1.tar.gz
cd opencv_contrib-4.5.1

tar -zxvf opencv-4.5.1.tar.gz
cv opencv-4.5.1
mkdir build
cd build

# 查找gcc
whereis gcc
whereis cmake3
./cmake3 --version

# 指定gcc版本
export CC=/usr/local/bin/gcc
export CXX=/usr/local/bin/g++
cmake3 -D CMAKE_BUILD_TYPE=RELEASE \
 	  -D CMAKE_INSTALL_PREFIX=/usr/local \
 	  -D INSTALL_C_EXAMPLES=ON \
 	  -D INSTALL_PYTHON_EXAMPLES=ON \
 	  -D OPENCV_GENERATE_PKGCONFIG=ON \
 	  -D BUILD_EXAMPLES=ON ..
 	  
 	  
make -j8
sudo make install
```

### cuda库授权

```shell

sudo chmod 777 -R /usr/local/cuda/include
sudo chmod 777 -R /usr/local/cuda/lib64
```

### 转换tesorrt

```cpp
#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.1
#define CONF_THRESH 0.1
#define BATCH_SIZE 1

#define NET x  // s m l x
```

```c++
    static constexpr int CLASS_NUM = 5;
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;
```

> yolov5.cpp 对照 yolov5.yaml 进行修改网络结构

```shell
./yolov5-tensorrt -s
./yolov5-tensorrt -d ./images
```

