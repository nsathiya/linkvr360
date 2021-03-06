#include "MemoryManager.h"


MemoryManager::MemoryManager(int NO_OF_CAMS,std::vector<std::string> output, int frameWidth, int frameHeight, bool Video)
{
	//error checking

	videoWriters = std::vector<cv::VideoWriter>(NO_OF_CAMS);

	if (Video)
		for (int i=0; i < NO_OF_CAMS; i++){
			cv::VideoWriter outputVideo;
			outputVideo.open(output[i], -1, 20, cv::Size(frameWidth, frameHeight), true);
			videoWriters[i] = outputVideo;
		}
}


MemoryManager::~MemoryManager()
{
}

void MemoryManager::readFrames(std::vector<cv::Mat> &FRAMES, std::string baseFileName){

	for (int i = 0; i < FRAMES.size(); i++){
		std::string frameName = baseFileName + "_" + std::to_string(i) + ".png";
		FRAMES[i] = cv::imread(frameName);
	}
}

void MemoryManager::writeStaticFrames(std::vector<cv::Mat> &FRAMES, int NO_OF_FRAMES, std::string baseFileName){

	//error checking

	for (int i = 0; i < NO_OF_FRAMES; i++){
		std::string frameName = baseFileName + "_" + std::to_string(i) + ".png";
		cv::imwrite(frameName, FRAMES[i]);
	}

}

void MemoryManager::writeVideoFrames(std::vector<cv::Mat> &FRAMES){

	//error checking

	for (int i = 0; i < FRAMES.size(); i++){
		videoWriters[i] << FRAMES[i];
	}

}
