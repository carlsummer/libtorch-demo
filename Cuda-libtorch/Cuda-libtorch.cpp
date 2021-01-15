// Cuda-libtorch.cpp: 定义应用程序的入口点。
//

#include "Cuda-libtorch.h"
#include <torch/torch.h>
#include <torch/script.h>
using namespace std;

int main()
{
	cout << "Hello CMake." << endl;
	torch::Tensor tensor = torch::rand({ 2, 3 }).to(at::kCPU); // GPU .to(at::kCUDA)  CPU .to(at::kCPU)
	std::cout << tensor << std::endl;

	// set device type - CPU/GPU
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		device_type = torch::kCUDA;
	}
	else {
		device_type = torch::kCPU;
	}


	cout << "load model..." << endl;
	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load("C:\\Users\\zengxh\\Documents\\workspace\\visualstudio-workspace\\libtorch-demo\\Cuda-libtorch\\weights\\solarcell.torchscript.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "Error loading the model!\n";
		std::exit(EXIT_FAILURE);
	}

	module.eval();
	module.to(at::kCPU);


	int y = getchar();
	return 0;
}
