// torchdemo.cpp: 定义应用程序的入口点。
//

#include "torchdemo.h"
#include <torch/torch.h>
using namespace std;

int main()
{
	cout << "Hello CMake." << endl;
	torch::Tensor tensor = torch::rand({ 2, 3 });
	std::cout << tensor << std::endl;
	int y = getchar();
	return 0;
}
