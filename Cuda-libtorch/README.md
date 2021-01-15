# 使用libtorch 加载pytorch训练好的模型

### python的torch版本
```shell
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu torchaudio===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html


```

### 生成visual studio运行代码：

**下载libtorch**:https://pytorch.org/get-started/locally/

**下载cmake**：https://cmake.org/download/

**直接用vs2019打开**：file->cmake->选择文件夹
**使用cmake生成代码**：

![cmake2vs](.\cmake2vs.png)

### 训练好的模型转换
```python
import torch
import torchvision.models as models
import torch.nn as nn


# 载入预训练的型
model = models.squeezenet1_1(pretrained=False)
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
model.num_classes = 2
model.load_state_dict(torch.load('./model/model_squeezenet_utk_face_20.pth', map_location='cpu'))
model.eval()
print(model)

# model = models.resnet18(pretrained=True)
# print(model)

example = torch.rand(1, 3, 224, 224)

traced_script_module = torch.jit.trace(model, example)

output = traced_script_module(torch.ones(1, 3, 224, 224))
print(output)


# ----------------------------------
traced_script_module.save("./model/model_squeezenet_utkface.pt")

```

### cmakelists 命令说明
**add_subdirectory**: 一般情况下，我们的项目各个子项目都在一个总的项目根目录下，但有的时候，我们需要使用外部的文件夹，怎么办呢？ 
add_subdirectory命令，可以将指定的文件夹加到build任务列表中。下面是将与当前项目平级的一个目录下的子目录用

### 参考网址：
> https://github.com/yasenh/libtorch-yolov5#torchscript-model-export
> https://blog.csdn.net/weixin_42398658/article/details/111954760