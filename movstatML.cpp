// movstatML.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "../libml/serialization.h"
#include "../libml/Tensor.h"
#include "../libml/DeviceContext.h"
#include "../libml/Graph.h"
#include "../libml/GraphNode.h"
#include "../libml/GraphOptimizer.h"
#include "../libml/math_util.h"
#include "../libml/conv_util.h"
#include "../libml/tensor_op.h"
#include "../libml/Detector_v3.h"
#include "../libml/Detector_v4.h"
#include "../libml/Detector_v5.h"
#include "../libml/TileManager.h"
#include "../libml/Conv2D.h"

#include "opencv2/opencv.hpp"

#include <algorithm>
#include <chrono>
#include <string>
#include <fstream>

using namespace ml;

void import_graph()
{
	Graph graph;

	graph.importFromKeras("D:\\siamese_folder\\");

	GraphOptimizer opt(graph);
	opt.optimize();
	auto so_g = opt.getGraph().saveGraph();
	auto so_w = opt.getGraph().saveWeights();
	so_g.saveToFile("D:\\ml_v2_detector\\comparator_v3_graph.bin");
	so_w.saveToFile("D:\\ml_v2_detector\\comparator_v3_weight.bin");
	opt.getGraph().print();
}

Tensor load_test_image(const std::string &path)
{
	std::ifstream f(path);
	std::string tmp;

	std::getline(f, tmp);
	int height = std::atoi(tmp.c_str());
	std::getline(f, tmp);
	int width = std::atoi(tmp.c_str());
	std::getline(f, tmp);
	int channels = std::atoi(tmp.c_str());

	Tensor result({ 1, height, width, channels }, DataType::UINT8, Device::cpu());
	for (int i = 0; i < result.volume(); i++)
	{
		std::getline(f, tmp);
		result.data<unsigned char>()[i] = std::atoi(tmp.c_str());
	}
	return result;
}

void test_speed()
{
	Tensor t_cpu({ 32, 128, 128, 128 }, DataType::FLOAT32, Device::cpu());
	Tensor t_gpu({ 32, 128, 128, 128 }, DataType::FLOAT32, Device::cuda(0));
	t_cpu.pageLock();

	Device::cpu().setNumberOfThreads(1);
	Graph graph;
	SerializedObject so_g("D:\\oid_training\\opt_graph.bin");
	SerializedObject so_w("D:\\oid_training\\opt_weight.bin");
	graph.loadGraph(so_g);
	graph.setInputShape({ 0, 256, 448, 3 });
	graph.loadWeights(so_w);
	DeviceContext cxt(Device::cuda(0));

	graph.forward(graph.getMaxBatchSize());

	graph.synchronize();

	for (int i = 0; i < graph.getNumberOfOutputs(); i++)
		std::cout << i << " : " << ml::norm(graph.getOutput(i)) << std::endl;;
}

void test_processing()
{
	Detector_v5 detector({ 1024, 256 }, "D:\\ml_v2_detector\\person_v5", Device::cpu());

	Tensor img = load_test_image("D:\\obrazek.txt");
	detector.getInput().copyFrom(img);
	cv::Mat img2 = cv::imread("D:\\dataset\\images\\00000033.jpg");
	cv::Mat res_img(256, 1024, CV_8UC3);
	cv::resize(img2, res_img, cv::Size(1024, 256));

	std::cout << std::setprecision(12);
	float min_score = 0.5f;
	float min_iou = 0.3f;
	bool use_soft_nms = false;

	//detector.detectOnImage(min_score);
	//return;
	std::cout << "---------------------------------------" << std::endl;

	auto start = std::chrono::system_clock::now();

	auto boxes = detector.detectOnImage(min_score);
	std::cout << boxes.size() << std::endl;
	applyNMS(boxes, min_score, min_iou, use_soft_nms);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	double end_time = (double)std::chrono::system_clock::to_time_t(end);
	std::cout << "total time: " << 1000 * elapsed_seconds.count() << "ms\n";

	std::cout << boxes.size() << std::endl;

	for (auto i = 0; i < boxes.size(); i++)
		std::cout << boxes[i].toString() << std::endl;

	drawBoxes(res_img, boxes);
	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
	cv::imshow("image", res_img);
	cv::waitKey(0);
}

cv::Mat tensor_to_mat(const Tensor &t)
{
	assert(t.dim() == 4);
	assert(t.shape(0) == 1);
	assert(t.dtype() == DataType::FLOAT32);
	cv::Mat result(t.shape(1), t.shape(2), CV_8UC3);
	for (int i = 0; i < t.volume(); i++)
		result.data[i] = static_cast<uint8_t>(255.0f * t.data<float>()[i]);
	return result;
}

Tensor mat_to_tensor(const cv::Mat &m)
{
	Tensor result({ 1, m.rows, m.cols, 3 }, DataType::FLOAT32, Device::cpu());
	for (int i = 0; i < result.volume(); i++)
		result.data<float>()[i] = static_cast<float>(m.data[i]) / 255.0f;
	return result;
}

void test_tiling()
{
	Detector_v5 detector({ 1024, 256 }, "D:\\ml_v2_detector\\person_v5", Device::cpu());
	TileManager tiler({ 256, 1024 }, 0.05f, 32, Device::cpu());

	cv::Mat img = cv::imread("D:\\dataset\\images\\00000033.jpg");
	cv::Mat res_img(256, 1024, CV_8UC3);
	cv::resize(img, res_img, cv::Size(1024, 256));
	std::memcpy(detector.getInput().data(), res_img.data, 256 * 1024 * 3);

	std::cout << std::setprecision(12);

	auto boxes = detector.detectOnImage(0.5f);
	applyNMS(boxes, 0.5f, 0.3f, false);
	std::cout << boxes.size() << std::endl;
	for (auto i = 0; i < boxes.size(); i++)
		std::cout << boxes[i].toString() << std::endl;
	drawBoxes(res_img, boxes);
	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
	cv::imshow("image", res_img);
	cv::waitKey(0);
}

std::string parse_time(double t)
{
	int h = int(t) / 3600;
	int m = int((t - h * 3600) / 60);
	int s = int(t) % 60;
	std::string result;
	if (h < 10)
		result += '0';
	result += std::to_string(h) + ':';
	if (m < 10)
	{
		result += '0';
		result += std::to_string(m) + ':';
		if (s < 10)
		{
			result += '0';
			result += std::to_string(s);
		}
	}
	return result;
}

void process_video()
{
	std::cout << "process video" << std::endl;
	cv::VideoCapture cap("D:\\19003 - Cam 7 - Queue Length Analytics-108.170.108.235-20141029.120943-20141029.123622.mp4");
	std::cout << cap.isOpened() << std::endl;
	cv::VideoWriter out;
	std::unique_ptr<Detector_v5> detector;
	std::unique_ptr<TileManager> tiler;

	int image_size = 256;
	double time = 0.0;
	int frame_count = 0;
	while (true)
	{
		cv::Mat frame;
		cap.read(frame);
		if (frame.empty())
			break;

		cv::Mat res_img(image_size, frame.cols * image_size / frame.rows, CV_8UC3);
		cv::resize(frame, res_img, res_img.size());

		if (detector == nullptr)
		{
			detector = std::make_unique<Detector_v5>(Shape({ res_img.cols, res_img.rows }),
				"C:\\Users\\source\\repos\\movstatML\\person_network", Device::cuda(0));

			std::cout << "frame shape      " << Shape({ frame.rows, frame.cols, 3 }) << std::endl;
			std::cout << "processing shape " << detector->getInput().shape() << std::endl;
		}

		auto start = std::chrono::system_clock::now();
		std::memcpy(detector->getInput().data(), res_img.data, detector->getInput().volume());
		auto boxes = detector->detectOnImage(0.5f);
		applyNMS(boxes, 0.4f, 0.3f, true);
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		frame_count++;
		time += elapsed_seconds.count();
		if (frame_count % 100 == 0)
			std::cout << "fps: " << frame_count / time << std::endl;

		drawBoxes(res_img, boxes);
		int tmp;
		cv::Size size = cv::getTextSize(std::to_string(boxes.size()) + " objects", cv::FONT_HERSHEY_DUPLEX, 0.5, 1, &tmp);
		cv::rectangle(res_img, { 0, 0 }, { size.width + 2, 2 * size.height + 5 }, cv::Scalar(0, 0, 0), -1);
		cv::putText(res_img, std::to_string(boxes.size()) + " objects", { 1, size.height - 1 }, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
		cv::putText(res_img, std::to_string((int)(1.0 / elapsed_seconds.count() + 0.5)) + " fps", { 1, 2 * size.height + 2 }, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

		if (frame_count == 1)
			out = cv::VideoWriter("C:\\Users\\Documents\\scp\\refs\\ml_runtime\\gpu_queue.wmv", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12, res_img.size());
		out.write(res_img);
		imshow("Frame", res_img);
		char c = cv::waitKey(1);
		if (c == 27)
			break;
		if (c == ' ')
		{
			while (cv::waitKey() != ' ');
		}
	}
	cap.release();
	out.release();
	cv::destroyAllWindows();
	detector->print();
}

void compare_detection()
{
	// 0.0 fps: 4.04421 18000/18000
	// 0.025 fps : 25.8677 10593 / 18000
	// 0.05 fps: 31.4321 9329 / 18000
	// 0.1 fps: 38.8246 7874/18000
	// 0.2 fps: 48.5883 7926/18000

	cv::VideoCapture cap("D:\\BK888-Outdoor Front Entance-RAO-BK888-20200410.113000-20200410.120000.mp4");

	std::unique_ptr<Detector_v5> detector_opt;
	std::unique_ptr<Detector_v5> detector_base;

	int image_size = 256;
	double time = 0.0;
	int frame_count = 0;
	int correct = 0;
	while (true)
	{
		cv::Mat frame;
		cap.read(frame);
		if (frame.empty())
			break;

		cv::Mat res_img(image_size, frame.cols * image_size / frame.rows, CV_8UC3);
		cv::resize(frame, res_img, res_img.size());

		if (detector_opt == nullptr)
			detector_opt = std::make_unique<Detector_v5>(Shape({ res_img.cols, res_img.rows }),
				"D:\\ml_v2_detector\\car_v5", Device::cpu());
		if (detector_base == nullptr)
			detector_base = std::make_unique<Detector_v5>(Shape({ res_img.cols, res_img.rows }),
				"D:\\ml_v2_detector\\car_v5", Device::cuda(0));

		auto start = std::chrono::system_clock::now();
		std::memcpy(detector_opt->getInput().data(), res_img.data, detector_opt->getInput().volume());
		auto boxes_opt = detector_opt->detectOnImage(0.5f);
		applyNMS(boxes_opt, 0.5f, 0.3f, true);
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		std::memcpy(detector_base->getInput().data(), res_img.data, detector_base->getInput().volume());
		auto boxes_base = detector_base->detectOnImage(0.5f);
		applyNMS(boxes_base, 0.5f, 0.3f, true);

		correct += (int)(boxes_opt.size() == boxes_base.size());
		frame_count++;
		time += elapsed_seconds.count();
		if (frame_count % 100 == 0)
			std::cout << "fps: " << frame_count / time << " " << correct << "/" << frame_count << std::endl;
		//std::cout << 1000 * time / frame_count << " ms" << std::endl;

		cv::Mat opt_img = res_img.clone();
		drawBoxes(opt_img, boxes_opt);
		imshow("Frame_opt", opt_img);

		cv::Mat base_img = res_img.clone();
		drawBoxes(base_img, boxes_base);
		imshow("Frame_base", base_img);
		char c = cv::waitKey(1);
		if (c == 27)
			break;
		if (c == ' ')
		{
			while (cv::waitKey() != ' ');
		}
	}
	cap.release();
	cv::destroyAllWindows();
}

int main()
{
	std::cout << Device::hardwareInfo() << std::endl;

	process_video();
	return 0;

	Detector_v5 detector({ 448, 256 }, "C:\\Users\\source\\repos\\movstatML\\person_network", Device::cpu());

	cv::Mat loaded = cv::imread("D:\\movstat_dataset\\images\\00002998.jpg");
	cv::Mat img0(256, 448, CV_8UC3);
	cv::resize(loaded, img0, img0.size());

	std::memcpy(detector.getInput().data(), img0.data, detector.getInput().volume());

	// Ten krok wykona zar�wno detekcj� jak i ekstrakcj� cech z obrazu,
	// ale nie przypisze ich automatycznie do wykrytych box�w
	auto boxes = detector.detectOnImage(0.5f);

	// ten krok bez zmian
	applyNMS(boxes, 0.4f, 0.3f, true);

	// Dopiero metoda 'assignFeatures' wylicza wektor cech i przypisuje go do danego boxa.
	// Rozdzieli�em to, bo nie ma sensu liczy� tego dla box�w, kt�re i tak by odpad�y w trakcie NMS.
	for (size_t i = 0; i < boxes.size(); i++)
		detector.assignFeatures(boxes[i]);

	// Ta funkcja zwraca miar� podobie�stwa cech box�w w zakresie [0, 1].
	// 0 - w og�le nie podobne do siebie, 1 - identyczne
	// Je�eli nie da si� por�wna� dw�ch box�w (np. kt�ry� nie ma przypisanych cech) funkcja zwraca 0.
	// Sugerowana warto�� progu, od kt�rego uznajemy, �e boxy przedstawiaj� ten sam obiekt to 0.85.
	std::cout << compareFeatures(boxes[0], boxes[1]) << std::endl;

	drawBoxes(img0, boxes);
	imshow("Frame", img0);
	cv::waitKey(-1);
}


