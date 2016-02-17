#include "CameraOps.h"


CameraOps::CameraOps(std::vector<int> cameraPorts, std::vector<std::string> videoPorts, bool VIDEO)
{
	
	if (VIDEO){
		for (auto i = 0; i < videoPorts.size(); i++) {
			cv::VideoCapture newCamera(videoPorts[i]);
			cameraArray.push_back(newCamera);
		}
	} else {
		for (auto i = 0; i < cameraPorts.size(); i++) {
			cv::VideoCapture newCamera(cameraPorts[i]);
			cameraArray.push_back(newCamera);
		}
	}
}

CameraOps::~CameraOps()
{
}

void CameraOps::CO_setProp(int prop, double value) {
	
	for (auto i = 0; i < cameraArray.size(); i++){
		cameraArray[i].set(prop, value);
	}
}

double CameraOps::CO_getProp(int prop, int camera) {

	return cameraArray[camera].get(prop);
}

void CameraOps::CO_captureFrames(std::vector<cv::Mat> &FRAMES){
	for (auto i = 0; i < cameraArray.size(); i++){
		cameraArray[i] >> FRAMES[i];
	}
}