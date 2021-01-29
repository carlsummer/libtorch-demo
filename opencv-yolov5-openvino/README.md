### 下载openvino
> https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html

### 导出yolov5 onnx 
> export-cpu.py
> 参考 https://blog.csdn.net/weixin_44936889/article/details/110940322
> 预训onnx模型demo_onnx.py

### 安装openvino
> https://docs.openvinotoolkit.org/2019_R2/_docs_install_guides_installing_openvino_windows.html

### 使用
> https://docs.openvinotoolkit.org/2021.2/openvino_docs_install_guides_installing_openvino_windows.html#set-the-environment-variables
> 设置环境变量
> cd C:\Program Files (x86)\Intel\openvino_2021\bin\
> setupvars.bat
> 安装依赖
> C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\install_prerequisites
> .\install_prerequisites.bat
> 运行demo
> cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo
> .\demo_squeezenet_download_convert_run.bat  # 需要翻墙访问下能下载
> 运行demo例子
> .\demo_security_barrier_camera.bat # 可以看到car.png被标注了
> 打包好的demo项目路径
> C:\Users\zengxh\Documents\Intel\OpenVINO\inference_engine_demos_build\intel64\Release
> cpu 运行打包好的项目
>  .\security_barrier_camera_demo.exe -i "C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo\car.png" -m "C:\Users\zengxh\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1\FP16\squeezenet1.1.xml" -d CPU
> GPU运行打包好的项目
> .\security_barrier_camera_demo.exe -i "C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo\car.png" -m "C:\Users\zengxh\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1\FP16\squeezenet1.1.xml" -d GPU

### 将yolov5的onnx 转换为openvino
> 将打包好的onnx copy到C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer
> 激活openvino的环境变量
> # conda activate AI
> cd C:\Program Files (x86)\Intel\openvino_2021\bin\
> .\setupvars.bat
> 安装好依赖
> cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer
> pip install -r requirements_onnx.txt
> 运行模型转换脚本
> cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer
> mo.py的转换参数 https://docs.openvinotoolkit.org/cn/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html
> python mo.py --input_model solarcell.onnx --output_dir C:\Users\zengxh\Desktop --input_shape [1,3,640,640] --data_type FP16
> python mo_onnx.py --input_model C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino\weights\solarcell.onnx --output_dir C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino\weights --input_shape [1,3,640,640] --data_type FP32
> 使用
> cd C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino
> python openvino-yolov5.py --model_xml="C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino\weights\solarcell.xml" --source="C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino\images"


