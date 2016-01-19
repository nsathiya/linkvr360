#include "ImageOps.h"


ImageOps::ImageOps()
{
}


ImageOps::~ImageOps()
{
}

/***************************
//
// Public ImageOps Functions
//
/****************************/

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
	if (intrinsicCoeffs.size() != imagesArray.size())
		throw new std::exception("Each Image index should have an intrinsicCoeffs");
	if (distortionCoeffs.size() != imagesArray.size())
		throw new std::exception("Each Image index should have an distortionCoeffs");


	for (auto i = 0; i < imagesArray.size(); i++) {
		cv::Mat result = cv::Mat(imagesArray[i].rows, imagesArray[i].cols, imagesArray[i].type());
		cv::undistort(imagesArray[i], result, intrinsicCoeffs[i], distortionCoeffs[i]);
		imagesArray[i] = result;
	}
}

void ImageOps::IO_perspectiveTransform(std::vector<cv::Mat> &imagesArray, std::vector<cv::Mat> externalCoeffs){
	
	if (imagesArray.size() == 0 || &imagesArray == NULL)
		throw new std::exception("Please check imagesArray input parameter");
	if (externalCoeffs.size() != imagesArray.size())
		throw new std::exception("Each image index should have an externalCoeffs value");
	
	for (auto i = 0; i < imagesArray.size(); i++) {
		cv::perspectiveTransform(imagesArray[i], imagesArray[i], externalCoeffs[i]);
	}
}

void ImageOps::IO_rectilinearProject(std::vector<cv::Mat> &imagesArray, int INV_FLAG, std::vector<float> FOCAL){

	if (imagesArray.size() == 0 || &imagesArray == NULL)
		throw new std::exception("Please check imagesArray input parameter");
	if (FOCAL.size() != imagesArray.size())
		throw new std::exception("Each image index should have a focal value");

	for (auto i = 0; i < imagesArray.size(); i++) {
		cv::Mat result;
		result = rectlinearProject(imagesArray[i], INV_FLAG, FOCAL[i]);
		imagesArray[i] = result;
	}
}

void ImageOps::IO_warpPerspective(std::vector<cv::Mat> &imagesArray, std::vector<cv::Mat> &resultsArray, std::vector<cv::Mat> externalCoeffs, cv::Size dSize){
	
	if (imagesArray.size() == 0 || &imagesArray == NULL)
		throw new std::exception("Please check imagesArray input parameter");
	if (resultsArray.size() != imagesArray.size())
		throw new std::exception("Each image index should have results Mat index");
	if (externalCoeffs.size() != imagesArray.size())
		throw new std::exception("Each image index should have an extrenalCoeffs value");

	for (auto i = 0; i < imagesArray.size(); i++) {
		cv::warpPerspective(imagesArray[i], resultsArray[i], externalCoeffs[i], dSize, cv::INTER_NEAREST | CV_WARP_FILL_OUTLIERS);
	}
}


/***************************
//
// Private Helper Functions
//
/****************************/

template<typename T>
void ImageOps::interpolateBilinear(cv::Mat Pic, cv::Point2f current_pos, cv::Point2i topLeftPoint, T &value)
{
	float dx = current_pos.x - topLeftPoint.x;
	float dy = current_pos.y - topLeftPoint.y;

	float weight_tl = (1.0f - dx) * (1.0f - dy);
	float weight_tr = (dx)* (1.0f - dy);
	float weight_bl = (1.0f - dx) * (dy);
	float weight_br = (dx)* (dy);

	value = weight_tl * Pic.at<T>(topLeftPoint) +
		weight_tr * Pic.at<T>(topLeftPoint.y, topLeftPoint.x + 1) +
		weight_bl * Pic.at<T>(topLeftPoint.y + 1, topLeftPoint.x) +
		weight_br * Pic.at<T>(topLeftPoint.y + 1, topLeftPoint.x + 1);
}

cv::Mat ImageOps::rectlinearProject(cv::Mat ImgToCalibrate, bool INV_FLAG, float F)
{
	cv::Mat Img = ImgToCalibrate;
	int height = Img.rows;
	int width = Img.cols;
	cv::Mat destPic = cv::Mat(cv::Size(width, height), ImgToCalibrate.type());
	std::cout << "rect linear " << ImgToCalibrate.type() << " " << CV_8UC3 << " " << CV_8UC1 << " " << CV_8UC4 << std::endl;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			cv::Point2f current_pos(x, y);
			current_pos = convert_pt(current_pos, width, height, INV_FLAG, F);

			cv::Point2i top_left((int)current_pos.x, (int)current_pos.y); //top left because of integer rounding

			//make sure the point is actually inside the original image
			if (top_left.x < 0 ||
				top_left.x > width - 2 ||
				top_left.y < 0 ||
				top_left.y > height - 2)
			{
				continue;
			}
			if (destPic.type() == CV_8UC1) {
				interpolateBilinear(Img, current_pos, top_left, destPic.at<uchar>(y, x));
			}
			else {//JH: added color pixels
				interpolateBilinear(Img, current_pos, top_left, destPic.at<cv::Vec3b>(y, x));
			}

		}
	}
	return destPic;
}

cv::Point2f ImageOps::convert_pt(cv::Point2f point, int w, int h, int INV_FLAG, float F)
{
	//center the point at 0,0
	cv::Point2f pc(point.x - w / 2, point.y - h / 2);
	float f, r;
	if (INV_FLAG == 0)
	{
		f = -F;
		r = w + 25;
	}
	else
	{
		f = 1;
		r = 1;
	}
	float omega = w / 2;
	float z0 = f - sqrt(r*r - omega*omega);

	float zc = (2 * z0 + sqrt(4 * z0*z0 - 4 * (pc.x*pc.x / (f*f) + 1)*(z0*z0 - r*r))) / (2 * (pc.x*pc.x / (f*f) + 1));
	cv::Point2f final_point(pc.x*zc / f, pc.y*zc / f);
	final_point.x += w / 2;
	final_point.y += h / 2;
	return final_point;
}