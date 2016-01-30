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
#include <opencv2/gpu/gpu.hpp>

class GPUOps
{
public:
	GPUOps(int NO_OF_CAMS);
	~GPUOps();
	void GO_uploadStream(std::vector<cv::Mat> &FRAMES);
	void GO_perspectiveTransform(std::vector<cv::Mat> externalCoeffs, int resultHeight, int resultWidth);
	void GPUOps::GO_downloadStream(std::vector<cv::Mat> &RESULTS);

private:
	std::vector<cv::gpu::GpuMat> IMAGEFRAMES;
	std::vector<cv::gpu::GpuMat> RESULTFRAMES;
	std::vector<cv::gpu::Stream> streams;
};

