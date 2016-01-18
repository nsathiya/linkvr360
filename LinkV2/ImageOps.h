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

class ImageOps
{
public:
	ImageOps();
	~ImageOps();

	void IO_resize(std::vector<cv::Mat> &imagesArray, cv::Size dSize);
	void IO_transpose(std::vector<cv::Mat> &imagesArray);
	void IO_flip(std::vector<cv::Mat> &imagesArray, int flipCode);
	void IO_cvtColor(std::vector<cv::Mat> &imagesArray, int colorCode);
	void IO_undistort(std::vector<cv::Mat> &imagesArray, std::vector<cv::Mat> intrinsicCoeffs,
		std::vector<cv::Mat> distortionCoeffs);
	void IO_perspectiveTransform(std::vector<cv::Mat> &imagesArray, std::vector<cv::Mat> externalCoeffs);
	void IO_rectilinearProject(std::vector<cv::Mat> &imagesArray, int INV_FLAG, std::vector<float> FOCAL);
	void IO_warpPerspective(std::vector<cv::Mat> &imagesArray, std::vector<cv::Mat> externalCoeffs, cv::Size dSize);
	
};

