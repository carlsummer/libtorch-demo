### ����openvino
> https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html

### ����yolov5 onnx 
> export-cpu.py
> �ο� https://blog.csdn.net/weixin_44936889/article/details/110940322
> Ԥѵonnxģ��demo_onnx.py

### ��װopenvino
> https://docs.openvinotoolkit.org/2019_R2/_docs_install_guides_installing_openvino_windows.html

### ʹ��
> https://docs.openvinotoolkit.org/2021.2/openvino_docs_install_guides_installing_openvino_windows.html#set-the-environment-variables
> ���û�������
> cd C:\Program Files (x86)\Intel\openvino_2021\bin\
> setupvars.bat
> ��װ����
> C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\install_prerequisites
> .\install_prerequisites.bat
> ����demo
> cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo
> .\demo_squeezenet_download_convert_run.bat  # ��Ҫ��ǽ������������
> ����demo����
> .\demo_security_barrier_camera.bat # ���Կ���car.png����ע��
> ����õ�demo��Ŀ·��
> C:\Users\zengxh\Documents\Intel\OpenVINO\inference_engine_demos_build\intel64\Release
> cpu ���д���õ���Ŀ
>  .\security_barrier_camera_demo.exe -i "C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo\car.png" -m "C:\Users\zengxh\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1\FP16\squeezenet1.1.xml" -d CPU
> GPU���д���õ���Ŀ
> .\security_barrier_camera_demo.exe -i "C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo\car.png" -m "C:\Users\zengxh\Documents\Intel\OpenVINO\openvino_models\ir\public\squeezenet1.1\FP16\squeezenet1.1.xml" -d GPU

### ��yolov5��onnx ת��Ϊopenvino
> ������õ�onnx copy��C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer
> ����openvino�Ļ�������
> # conda activate AI
> cd C:\Program Files (x86)\Intel\openvino_2021\bin\
> .\setupvars.bat
> ��װ������
> cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer
> pip install -r requirements_onnx.txt
> ����ģ��ת���ű�
> cd C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer
> mo.py��ת������ https://docs.openvinotoolkit.org/cn/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html
> python mo.py --input_model solarcell.onnx --output_dir C:\Users\zengxh\Desktop --input_shape [1,3,640,640] --data_type FP16
> python mo_onnx.py --input_model C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino\weights\solarcell.onnx --output_dir C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino\weights --input_shape [1,3,640,640] --data_type FP32
> ʹ��
> cd C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino
> python openvino-yolov5.py --model_xml="C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino\weights\solarcell.xml" --source="C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\libtorch-yolov5-openvino\images"


