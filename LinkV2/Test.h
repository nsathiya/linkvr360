#pragma once
#include <iostream>
#include <fstream>
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
#include "ImageOps.h"
#include "CameraOps.h"
#include "GPUOps.h"
#include "MemoryManager.h"
#include "BlenderOps.h"

#include <stdio.h>  /* defines FILENAME_MAX */
#include <direct.h>
#define GetCurrentDir _getcwd


class Test
{
public:
	Test(bool showPic);
	~Test();
	int getWorld();

private:
	int testingFunction(bool GPU, bool stitchFromMemory, bool stitchVideo);
	cv::Mat border(cv::Mat mask);
	void setup();
	bool testPic;
	

	
};

