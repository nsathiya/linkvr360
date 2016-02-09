#include "GPUOps.h"


GPUOps::GPUOps(int NO_OF_CAMS)
{
	cv::gpu::setDevice(0);
	streams = std::vector<cv::gpu::Stream>(NO_OF_CAMS);
	IMAGEFRAMES = std::vector<cv::gpu::GpuMat>(NO_OF_CAMS);
	RESULTFRAMES = std::vector<cv::gpu::GpuMat>(NO_OF_CAMS);

	for (int i = 0; i < streams.size(); i++){
		cv::gpu::GpuMat image;
		cv::gpu::GpuMat result;
		IMAGEFRAMES[i] = image;
		RESULTFRAMES[i] = result;
	}
}


GPUOps::~GPUOps()
{

}

void GPUOps::GO_uploadStream(std::vector<cv::Mat> &FRAMES){
	//error statements

	for (int i = 0; i < streams.size(); i++){
		cv::gpu::GpuMat imageSrc;
		streams[i].enqueueUpload(FRAMES[i], IMAGEFRAMES[i]);
	}
}

void GPUOps::GO_warpPerspective(std::vector<cv::Mat> externalCoeffs, int resultHeight, int resultWidth){
	//error statements

	for (int i = 0; i < streams.size(); i++){
		cv::gpu::warpPerspective(IMAGEFRAMES[i], RESULTFRAMES[i], externalCoeffs[i], cv::Size(resultHeight, resultWidth), 1, 0, cv::Scalar(), streams[i]);
		streams[i].waitForCompletion();
	}
}

void GPUOps::GO_downloadStream(std::vector<cv::Mat> &RESULTS){
	//error statements

	for (int i = 0; i < streams.size(); i++){
		streams[i].enqueueDownload(RESULTFRAMES[i], RESULTS[i]);
	}
}



