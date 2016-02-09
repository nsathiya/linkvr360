#pragma once
#include <iostream>
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/nonfree/nonfree.hpp"

class MemoryManager
{
public:
	MemoryManager(int NO_OF_CAMS, std::vector<std::string> output, int frameHeight, int frameWidth);
	~MemoryManager();

	void readFrames(std::vector<cv::Mat> &FRAMES, std::string baseFileName);
	void writeStaticFrames(std::vector<cv::Mat> &FRAMES, int NO_OF_FRAMES, std::string baseFileName);
	void writeVideoFrames(std::vector<cv::Mat> &FRAMES);

private:
	std::vector<cv::VideoWriter> videoWriters;

};

