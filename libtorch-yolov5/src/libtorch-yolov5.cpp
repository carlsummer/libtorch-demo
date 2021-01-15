// libtorch-yolov5.cpp : Defines the entry point for the application.
//
#include <iostream>
#include <memory>
#include <chrono>

#include "libtorch-yolov5.h"
#include "detector.h"

using namespace std;

std::vector<std::string> LoadNames(const std::string& path) {
	// load class names
	std::vector<std::string> class_names;
	std::ifstream infile(path);
	if (infile.is_open()) {
		std::string line;
		while (getline(infile, line)) {
			class_names.emplace_back(line);
		}
		infile.close();
	}
	else {
		std::cerr << "Error loading the class names!\n";
	}

	return class_names;
}

void Demo(cv::Mat& img,
	const std::vector<std::vector<Detection>>& detections,
	const std::vector<std::string>& class_names,
	bool label = true) {

	if (!detections.empty()) {
		for (const auto& detection : detections[0]) {
			const auto& box = detection.bbox;
			float score = detection.score;
			int class_idx = detection.class_idx;

			cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

			if (label) {
				std::stringstream ss;
				ss << std::fixed << std::setprecision(2) << score;
				std::string s = class_names[class_idx] + " " + ss.str();

				auto font_face = cv::FONT_HERSHEY_DUPLEX;
				auto font_scale = 1.0;
				int thickness = 1;
				int baseline = 0;
				auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
				cv::rectangle(img,
					cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
					cv::Point(box.tl().x + s_size.width, box.tl().y),
					cv::Scalar(0, 0, 255), -1);
				cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
					font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
			}
		}
	}

	cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
	cv::imshow("Result", img);
	cv::waitKey(0);
}


int main()
{
	cout << "光伏组件外观检测" << endl;

	// set device tpye - cpu/GPU
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		device_type = torch::kCUDA;
	}
	else {
		device_type = torch::kCPU;
	}

	// load class names from dataset for visualization
	std::vector<std::string> class_names = LoadNames("./weights/coco.names");
	if (class_names.empty()) {
		return -1;
	}
	cout << class_names << endl;

	// load network
	std::string weights = "./weights/solarcell.torchscript.pt";
	auto detector = Detector(weights, device_type);

	// load input image
	std::string source = "C:\\Users\\zengxh\\Desktop\\images\\liepian.jpg";
	cv::Mat img = cv::imread(source);
	//cv::imshow("images", img);
	//cv::waitKey(0);
	if (img.empty()) {
		std::cerr << "Error loading the image!\n";
		return -1;
	}

	// run once to warm up
	std::cout << "Run once on empty image" << std::endl;
	auto temp_img = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
	detector.Run(temp_img, 1.0f, 1.0f);

	// set up threshold
	float conf_thres = 0.1;
	float iou_thres = 0.1;

	// inference
	auto result = detector.Run(img, conf_thres, iou_thres);

	// visualize detections
	Demo(img, result, class_names);

	cv::destroyAllWindows();
	return 0;
}
