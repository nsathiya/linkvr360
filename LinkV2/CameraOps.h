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

class CameraOps
{
public:
	CameraOps(std::vector<int> cameraPorts, std::vector<std::string> videoPorts, bool VIDEO);
	~CameraOps();
	void CO_setProp(int prop, double value);
	double CO_getProp(int prop, int camera);
	void CO_captureFrames(std::vector<cv::Mat> &FRAMES);

private:
	std::vector<cv::VideoCapture> cameraArray;
};

