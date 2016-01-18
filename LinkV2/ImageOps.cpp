#include "ImageOps.h"


ImageOps::ImageOps()
{
}


ImageOps::~ImageOps()
{
}

void ImageOps::IO_resize(std::vector<cv::Mat> &imagesArray, cv::Size dSize) {
	
	if (imagesArray.size() == 0 || &imagesArray == NULL)
		throw new std::exception("Please check imagesArray input parameter");
	if (!(dSize.height && dSize.width))
		throw new std::exception("Please check sizeToResize input parameter");
	
	for (auto i = 0; i < imagesArray.size(); i++)
	{	
		cv::resize(imagesArray[i], imagesArray[i], dSize);
	}
}

void ImageOps::IO_transpose(std::vector<cv::Mat> &imagesArray) {

	if (imagesArray.size() == 0 || &imagesArray == NULL)
		throw new std::exception("Please check imagesArray input parameter");

	for (auto i = 0; i < imagesArray.size(); i++) {
		cv::transpose(imagesArray[i], imagesArray[i]);
	}
}

void ImageOps::IO_flip(std::vector<cv::Mat> &imagesArray, int flipCode) {

	if (imagesArray.size() == 0 ||&imagesArray == NULL)
		throw new std::exception("Please check imagesArray input parameter");
	if (flipCode < 0 || flipCode > 5) //CHECK 
		throw new std::exception("Please input valid flip code");

	for (auto i = 0; i < imagesArray.size(); i++) {
		cv::flip(imagesArray[i], imagesArray[i], flipCode);
	}
}

void ImageOps::IO_cvtColor(std::vector<cv::Mat> &imagesArray, int colorCode){
	
	if (imagesArray.size() == 0 || &imagesArray == NULL)
		throw new std::exception("Please check imagesArray input parameter");

	for (auto i = 0; i < imagesArray.size(); i++) {
		cv::cvtColor(imagesArray[i], imagesArray[i], colorCode);
	}
}

void ImageOps::IO_undistort(std::vector<cv::Mat> &imagesArray, std::vector<cv::Mat> intrinsicCoeffs,
	std::vector<cv::Mat> distortionCoeffs){
	
	if (imagesArray.size() == 0 || &imagesArray == NULL)
		throw new std::exception("Please check imagesArray input parameter");
	if (intrinsicCoeffs.size() == 0 || &intrinsicCoeffs == NULL)
		throw new std::exception("Please check intrinsicCoeffs input parameter");
	if (distortionCoeffs.size() == 0 || &distortionCoeffs == NULL)
		throw new std::exception("Please check distortionCoeffs input parameter");


	for (auto i = 0; i < imagesArray.size(); i++) {
		cv::Mat result = cv::Mat(imagesArray[i].rows, imagesArray[i].cols, imagesArray[i].type());
		cv::undistort(imagesArray[i], result, intrinsicCoeffs[i], distortionCoeffs[i]);
		imagesArray[i] = result;
	}
}

void ImageOps::IO_perspectiveTransform(std::vector<cv::Mat> &imagesArray, std::vector<cv::Mat> externalCoeffs){
	//TODO
}

void ImageOps::IO_rectilinearProject(std::vector<cv::Mat> &imagesArray, int INV_FLAG, std::vector<float> FOCAL){
	//TODO
}

void ImageOps::IO_warpPerspective(std::vector<cv::Mat> &imagesArray, std::vector<cv::Mat> externalCoeffs, cv::Size dSize){
	//TODO
}