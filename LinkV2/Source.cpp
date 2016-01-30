
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
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

#include "ImageOps.h"
#include "CameraOps.h"
#include "GPUOps.h"

using namespace std;
using namespace cv;

const int BASE_CAM = 1; // 4; // 1; //0;
const int LEFT_CAM = 2; // 3; // 2; // 4; // 2;
const int RIGHT_CAM = 4; // 1; // 4; // 2; // 3; // 1;
const int FOUR_CAM = 0; // 0; // 4;
const int FIFTH_CAM = 3; // 2; // 3; // 2; // 3;
const int BACK_CAM = 5;
const int NO_OF_CAMS = 5;

string videoPath = "videos";
string calibrationPath = "calibration";
string internalCalibrationPath = calibrationPath + "/internal";
string externalCalibrationPath = calibrationPath + "/external";
string camLOutput = videoPath + "/Cam_L_Stream.avi";
string camROutput = videoPath + "/Cam_R_Stream.avi";
string camBOutput = videoPath + "/Cam_B_Stream.avi";
string cam4Output = videoPath + "/Cam_4_Stream.avi";
string cam5Output = videoPath + "/Cam_5_Stream.avi";
string camResultOutput = videoPath + "/Cam_Result_Stream.avi";

// Distortion
string H_File, IntrinsicBase_File, DistRight_File, IntrinsicRight_File, DistBase_File, IntrinsicLeft_File, DistLeft_File, IntrinsicFour_File, DistFour_File, IntrinsicFive_File, DistFive_File, IntrinsicSix_File, DistSix_File;
cv::Mat baseIntrinsic = Mat(3, 3, CV_32FC1);
cv::Mat rightIntrinsic = Mat(3, 3, CV_32FC1);
cv::Mat leftIntrinsic = Mat(3, 3, CV_32FC1);
cv::Mat fourIntrinsic = Mat(3, 3, CV_32FC1);
cv::Mat fiveIntrinsic = Mat(3, 3, CV_32FC1);
cv::Mat sixIntrinsic = Mat(3, 3, CV_32FC1);
cv::Mat baseDistCoeffs, rightDistCoeffs, leftDistCoeffs, fourDistCoeffs, fiveDistCoeffs, sixDistCoeffs;
cv::Mat HR = cv::Mat(3, 3, CV_32FC1);
cv::Mat HL = cv::Mat(3, 3, CV_32FC1);
cv::Mat H4 = cv::Mat(3, 3, CV_32FC1);
cv::Mat H5 = cv::Mat(3, 3, CV_32FC1);
cv::Mat H6 = cv::Mat(3, 3, CV_32FC1);
string H_R, H_L, H_4, H_5, H_6;
std::map<int, float> CAM_F_MAP;
std::vector<Point2f> baseImage;
std::vector<Point2f> rightImage;
cv::Mat rightFrame, baseFrame;

/// Control grayScale option
bool useGrayScale = false;

void setup();
int showFrames();
int stitch();
int record();
int stitchLive();
int stitchLiveWOGPU();
int stitchLiveWGPU();
int use360Camera();
int recordSimple();
int testingFunction();

cv::Point2f convert_pt(cv::Point2f point, int w, int h, int INV_FLAG, float F);
cv::Mat rectlinearProject(Mat ImgToCalibrate, bool INV_FLAG, float F);
int calibrateCamerasInternal(int cam);
int calibrateCamerasExternal(int baseCam, int sideCam);
std::vector<cv::Mat> FRAMES(NO_OF_CAMS);
std::vector<cv::Mat> INTRINSICCOEFFS(NO_OF_CAMS);
std::vector<cv::Mat> EXTRINSICCOEFFS(NO_OF_CAMS);
std::vector<cv::Mat> DISTORTIONCOEFFS(NO_OF_CAMS);
std::vector<cv::Mat> RESULTS(NO_OF_CAMS);
std::vector<float> FOCAL(NO_OF_CAMS);

int main() {

	cout << cv::getBuildInformation() << endl;
	setup();

	std::string MainMenu = "Welcome to Link's imaging module.\n"
		"Press 's' to Sitch Live.\n"
		"Press 'r' to Record and Stitch. \n"
		"Press 'e' to Record Only. \n"
		"Press 'f' to Show Test Frames.\n"
		"Press 'c' for External Calibration.\n"
		"Press 'i' for Internal Calibration.\n"
		"Press 'g' for B&W/Color mode switching.\n Current mode is Color. \n"
		"Press 'a' for Stitch Live without GPU. \n"
		"Press 'b' for Stitch Live with GPU \n"
		"Press 't' for testing function.\n";
	std::cout << MainMenu << std::endl;

	while (1)
	{
		char optionSelected;
		std::cin >> optionSelected;

		if (optionSelected == 's')
		{
			if (stitchLive() == 1)
				return 0;
		}
		if (optionSelected == 'a')
		{
			if (stitchLiveWOGPU() == 1)
				return 0;
		}
		if (optionSelected == 'b')
		{
			if (stitchLiveWGPU() == 1)
				return 0;
		}
		if (optionSelected == 'r')
		{
			if (use360Camera() == 1)
				return 0;
		}
		if (optionSelected == 'e')
		{
			if (recordSimple() == 1)
				return 0;
		}
		if (optionSelected == 'f')
		{
			if (showFrames() == 1)
				return 0;
		}
		else if (optionSelected == 'c')
		{
			if (calibrateCamerasExternal(RIGHT_CAM, FOUR_CAM) == 1)
				return 0;
		}
		else if (optionSelected == 'i')
		{
			if (calibrateCamerasInternal(FIFTH_CAM) == 1)
				return 0;
		}
		else if (optionSelected == 'g') {
			/// JH: Control grayScale option enable/ disable
			useGrayScale = !useGrayScale;
			std::cout << "Color Mode is " << (useGrayScale ? "B&W" : "Color") << std::endl;
		}
		else if (optionSelected == 't') {
			if (testingFunction() == 1)
				return 0;
		}
	}
}

void setup()
{

	FOCAL[0] = CAM_F_MAP[BASE_CAM] = 175.0214;
	FOCAL[1] = CAM_F_MAP[LEFT_CAM] = 171.9173;
	FOCAL[2] = CAM_F_MAP[RIGHT_CAM] = 171.575;
	FOCAL[3] = CAM_F_MAP[FOUR_CAM] = 176.9511;
	FOCAL[4] = CAM_F_MAP[FIFTH_CAM] = 175.2695;
	//CAM_F_MAP[BACK_CAM] = 175.2695;

	IntrinsicBase_File = internalCalibrationPath + "/intrinsic-base.txt";
	DistBase_File = internalCalibrationPath + "/distortion-base.txt";
	IntrinsicRight_File = internalCalibrationPath + "/intrinsic-right.txt";
	DistRight_File = internalCalibrationPath + "/distortion-right.txt";
	IntrinsicLeft_File = internalCalibrationPath + "/intrinsic-left.txt";
	DistLeft_File = internalCalibrationPath + "/distortion-left.txt";
	IntrinsicFour_File = internalCalibrationPath + "/intrinsic-four.txt";
	DistFour_File = internalCalibrationPath + "/distortion-four.txt";
	IntrinsicFive_File = internalCalibrationPath + "/intrinsic-five.txt";
	DistFive_File = internalCalibrationPath + "/distortion-five.txt";
	//IntrinsicSix_File = "intrinsic-six.txt";
	//DistSix_File = "distortion-six.txt";

	FileStorage file(IntrinsicBase_File, FileStorage::READ);
	file["Intrinsic Matrix"] >> baseIntrinsic;
	file.open(DistBase_File, FileStorage::READ);
	file["Distortion Matrix"] >> baseDistCoeffs;
	file.open(IntrinsicRight_File, FileStorage::READ);
	file["Intrinsic Matrix"] >> rightIntrinsic;
	file.open(DistRight_File, FileStorage::READ);
	file["Distortion Matrix"] >> rightDistCoeffs;
	file.open(IntrinsicLeft_File, FileStorage::READ);
	file["Intrinsic Matrix"] >> leftIntrinsic;
	file.open(DistLeft_File, FileStorage::READ);
	file["Distortion Matrix"] >> leftDistCoeffs;
	file.open(IntrinsicFour_File, FileStorage::READ);
	file["Intrinsic Matrix"] >> fourIntrinsic;
	file.open(DistFour_File, FileStorage::READ);
	file["Distortion Matrix"] >> fourDistCoeffs;
	file.open(IntrinsicFive_File, FileStorage::READ);
	file["Intrinsic Matrix"] >> fiveIntrinsic;
	file.open(DistFive_File, FileStorage::READ);
	file["Distortion Matrix"] >> fiveDistCoeffs;
	//file.open(IntrinsicSix_File, FileStorage::READ);
	//file["Intrinsic Matrix"] >> sixIntrinsic;
	//file.open(DistSix_File, FileStorage::READ);
	//file["Distortion Matrix"] >> sixDistCoeffs;

	H_R = externalCalibrationPath + "/H-right.txt";
	H_L = externalCalibrationPath + "/H-left.txt";
	H_4 = externalCalibrationPath + "/H-four.txt";
	H_5 = externalCalibrationPath + "/H-five.txt";
	//H_6 = "H-six.txt";

	//Find scene information 
	FileStorage fsr(H_R, FileStorage::READ);
	FileStorage fsl(H_L, FileStorage::READ);
	fsr["H Matrix"] >> HR;
	fsl["H Matrix"] >> HL;
	fsr.open(H_4, FileStorage::READ);
	fsr["H Matrix"] >> H4;
	fsr.open(H_5, FileStorage::READ);
	fsr["H Matrix"] >> H5;
	//fsr.open(H_6, FileStorage::READ);
	//fsr["H Matrix"] >> H6;

	HR.convertTo(HR, CV_32FC1);
	HL.convertTo(HL, CV_32FC1);
	H4.convertTo(H4, CV_32FC1);
	H5.convertTo(H5, CV_32FC1);
	//H6.convertTo(H6, CV_32FC1);

	INTRINSICCOEFFS[0] = baseIntrinsic;
	INTRINSICCOEFFS[1] = leftIntrinsic;
	INTRINSICCOEFFS[2] = rightIntrinsic;
	INTRINSICCOEFFS[3] = fourIntrinsic;
	INTRINSICCOEFFS[4] = fiveIntrinsic;

	DISTORTIONCOEFFS[0] = baseDistCoeffs;
	DISTORTIONCOEFFS[1] = leftDistCoeffs;
	DISTORTIONCOEFFS[2] = rightDistCoeffs;
	DISTORTIONCOEFFS[3] = fourDistCoeffs;
	DISTORTIONCOEFFS[4] = fiveDistCoeffs;
	
	EXTRINSICCOEFFS[0] = NULL;
	EXTRINSICCOEFFS[1] = HL;
	EXTRINSICCOEFFS[2] = HR;
	EXTRINSICCOEFFS[3] = H4;
	EXTRINSICCOEFFS[4] = H5;

}

void RightCallBackFunc2(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "For Right Image - position (" << float(x) << ", " << float(y) << ")" << endl;
		rightImage.push_back(cv::Point2f(float(x), float(y)));
		cv::circle(rightFrame, cv::Point2f(float(x), float(y)), 3, cv::Scalar(0, 255, 0));
		imshow("Side Image", rightFrame);
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
}

void BaseCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "For Base Image - position (" << float(x) << ", " << float(y) << ")" << endl;
		baseImage.push_back(cv::Point2f(float(x), float(y)));
		cv::circle(baseFrame, cv::Point2f(float(x), float(y)), 3, cv::Scalar(0, 255, 0));
		imshow("Base Image", baseFrame);
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	//else if ( event == EVENT_MOUSEMOVE )
	//{
	//    cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
	//}
}

int showFrames()
{
	std::vector<int> cameraPorts(NO_OF_CAMS);
	cameraPorts[0] = BASE_CAM;
	cameraPorts[1] = LEFT_CAM;
	cameraPorts[2] = RIGHT_CAM;
	cameraPorts[3] = FOUR_CAM;
	cameraPorts[4] = FIFTH_CAM;
	CameraOps *CO = new CameraOps(cameraPorts);
	CO->CO_setProp(CV_CAP_PROP_FRAME_WIDTH, 400);
	CO->CO_setProp(CV_CAP_PROP_FRAME_HEIGHT, 300);

	double Brightness;
	double Contrast;
	double Saturation;
	double Gain;

	Brightness = CO->CO_getProp(CV_CAP_PROP_BRIGHTNESS, 0);
	Contrast = CO->CO_getProp(CV_CAP_PROP_CONTRAST, 0);
	Saturation = CO->CO_getProp(CV_CAP_PROP_SATURATION, 0);
	Gain = CO->CO_getProp(CV_CAP_PROP_GAIN, 0);

	cout << "Brightness: " << Brightness;
	cout << "Contrast: " << Contrast;
	cout << "Saturation: " << Saturation;
	cout << "Gain: " << Gain;

	CO->CO_setProp(CV_CAP_PROP_BRIGHTNESS, Brightness);
	CO->CO_setProp(CV_CAP_PROP_CONTRAST, Contrast);
	CO->CO_setProp(CV_CAP_PROP_SATURATION, Saturation);
	CO->CO_setProp(CV_CAP_PROP_GAIN, Gain);

	int frameWidth = CO->CO_getProp(CV_CAP_PROP_FRAME_WIDTH, 0)*0.25;
	int frameHeight = CO->CO_getProp(CV_CAP_PROP_FRAME_HEIGHT, 0)*0.25;
	cv::Mat leftFrame, rightFrame, fourFrame, fifthFrame, sixFrame;

	while (1)
	{
		CO->CO_captureFrames(FRAMES);

		if (waitKey(100) == 110)
			break;

		imshow("base Frame",  FRAMES[0]);
		imshow("left Frame",  FRAMES[1]);
		imshow("right Frame", FRAMES[2]);
		imshow("four Frame",  FRAMES[3]);
		imshow("five Frame",  FRAMES[4]);
		//imshow("sixth Frame", sixFrame);
	}
	delete CO;
	return 1;
}

/// Changed function to work with custom pixel types 
template<typename T>
void interpolateBilinear(Mat Pic, cv::Point2f current_pos, cv::Point2i topLeftPoint, T &value)
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
cv::Mat rectlinearProject(Mat ImgToCalibrate, bool INV_FLAG, float F)
{
	Mat Img = ImgToCalibrate;
	int height = Img.rows;
	int width = Img.cols;
	Mat destPic = Mat(cv::Size(width, height), ImgToCalibrate.type());
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

cv::Point2f convert_pt(cv::Point2f point, int w, int h, int INV_FLAG, float F)
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


int stitchLive()
{

	return 1;
}


int use360Camera()
{
	if (record() == 1)
		stitch();

	return 1;

}

int testingFunction() {

	return 1;
}

int stitchLiveWGPU() {

	cout << "Stitching with GPU... \n" << endl;

	std::vector<int> cameraPorts(NO_OF_CAMS);
	cameraPorts[0] = BASE_CAM;
	cameraPorts[1] = LEFT_CAM;
	cameraPorts[2] = RIGHT_CAM;
	cameraPorts[3] = FOUR_CAM;
	cameraPorts[4] = FIFTH_CAM;
	CameraOps *CO = new CameraOps(cameraPorts);
	ImageOps *IO = new ImageOps();
	GPUOps *GO = new GPUOps(NO_OF_CAMS);

	cv::VideoWriter outputVideo;
	cv::Mat result;

	std::vector<cv::Point2f> scene_cornersLeft, scene_cornersRight, scene_cornersBase, scene_cornersFour, scene_cornersFive, scene_cornersSix, scene_corners;
	CO->CO_setProp(CV_CAP_PROP_FRAME_WIDTH, 1920);
	CO->CO_setProp(CV_CAP_PROP_FRAME_HEIGHT, 1080);

	double Brightness;
	double Contrast;
	double Saturation;
	double Gain;

	Brightness = CO->CO_getProp(CV_CAP_PROP_BRIGHTNESS, 0);
	Contrast = CO->CO_getProp(CV_CAP_PROP_CONTRAST, 0);
	Saturation = CO->CO_getProp(CV_CAP_PROP_SATURATION, 0);
	Gain = CO->CO_getProp(CV_CAP_PROP_GAIN, 0);

	cout << "Brightness: " << Brightness;
	cout << "Contrast: " << Contrast;
	cout << "Saturation: " << Saturation;
	cout << "Gain: " << Gain;

	CO->CO_setProp(CV_CAP_PROP_BRIGHTNESS, Brightness);
	CO->CO_setProp(CV_CAP_PROP_CONTRAST, Contrast);
	CO->CO_setProp(CV_CAP_PROP_SATURATION, Saturation);
	CO->CO_setProp(CV_CAP_PROP_GAIN, Gain);

	int frameWidth = CO->CO_getProp(CV_CAP_PROP_FRAME_WIDTH, 0)*0.25;
	int frameHeight = CO->CO_getProp(CV_CAP_PROP_FRAME_HEIGHT, 0)*0.25;
	int resultWidth = frameHeight * 2;
	int resultHeight = frameWidth + 100;
	bool record = false;
	result = Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);

	// Move Scene to the right by 100
	int x_offset = 500;
	float y_offset = 0.0;
	float z_offset = 100.0;
	float transdata[] = { 1.0, 0.0, x_offset, 0.0, 1.0, y_offset, 0.0, 0.0, 1.0 };
	cv::Mat trans(3, 3, CV_32FC1, transdata);
	cout << "HR: " << HR << endl;
	EXTRINSICCOEFFS[0] = trans;

	Mat HR_m = HR.clone();
	Mat HL_m = HL.clone();
	HR = trans * HR;
	HL = trans * HL;
	H4 = trans * HR_m * H4;
	H5 = trans * HL_m * H5;
	//H6 = trans * HL_m * H5 * H6;

	cout << "finished getting matrix" << endl;

	CO->CO_captureFrames(FRAMES);

	if (FRAMES[0].cols == 0) {
		cout << "Error reading file " << endl;
		return -1;
	}

	IO->IO_resize(FRAMES, cv::Size(frameWidth, frameHeight));
	IO->IO_transpose(FRAMES);
	IO->IO_flip(FRAMES, 1);
	IO->IO_undistort(FRAMES, INTRINSICCOEFFS, DISTORTIONCOEFFS);

	// Use the Homography Matrix to warp the images
	scene_corners.clear();
	scene_cornersBase.push_back(Point2f(0.0, 0.0));
	scene_cornersBase.push_back(Point2f(FRAMES[0].cols, 0.0));
	scene_cornersBase.push_back(Point2f(0.0, FRAMES[0].rows));
	scene_cornersBase.push_back(Point2f(FRAMES[0].cols, FRAMES[0].rows));
	scene_cornersLeft.push_back(Point2f(0.0, 0.0));
	scene_cornersLeft.push_back(Point2f(FRAMES[1].cols, 0.0));
	scene_cornersLeft.push_back(Point2f(0.0, FRAMES[1].rows));
	scene_cornersLeft.push_back(Point2f(FRAMES[1].cols, FRAMES[1].rows));
	scene_cornersRight.push_back(Point2f(0.0, 0.0));
	scene_cornersRight.push_back(Point2f(FRAMES[2].cols, 0.0));
	scene_cornersRight.push_back(Point2f(0.0, FRAMES[2].rows));
	scene_cornersRight.push_back(Point2f(FRAMES[2].cols, FRAMES[2].rows));
	scene_cornersFour.push_back(Point2f(0.0, 0.0));
	scene_cornersFour.push_back(Point2f(FRAMES[3].cols, 0.0));
	scene_cornersFour.push_back(Point2f(0.0, FRAMES[3].rows));
	scene_cornersFour.push_back(Point2f(FRAMES[3].cols, FRAMES[3].rows));
	scene_cornersFive.push_back(Point2f(0.0, 0.0));
	scene_cornersFive.push_back(Point2f(FRAMES[4].cols, 0.0));
	scene_cornersFive.push_back(Point2f(0.0, FRAMES[4].rows));
	scene_cornersFive.push_back(Point2f(FRAMES[4].cols, FRAMES[4].rows));
	//scene_cornersSix.push_back(Point2f(0.0, 0.0));
	//scene_cornersSix.push_back(Point2f(sixFrame.cols, 0.0));
	//scene_cornersSix.push_back(Point2f(0.0, sixFrame.rows));
	//scene_cornersSix.push_back(Point2f(sixFrame.cols, sixFrame.rows));

	perspectiveTransform(scene_cornersBase, scene_cornersBase, trans);
	perspectiveTransform(scene_cornersLeft, scene_cornersLeft, HL);
	perspectiveTransform(scene_cornersRight, scene_cornersRight, HR);
	perspectiveTransform(scene_cornersFour, scene_cornersFour, H4);
	perspectiveTransform(scene_cornersFive, scene_cornersFive, H5);
	//perspectiveTransform(scene_cornersSix, scene_cornersSix, H6);


	//Store useful information for Image limits
	int leftLimit, baseLeftLimit, baseRightLimit, rightLimit, fourLimit, fifthLimit, sixLimit;

	fifthLimit = scene_cornersFive[1].x;
	//sixLimit = scene_cornersSix[1].x;
	leftLimit = scene_cornersLeft[1].x;
	baseLeftLimit = x_offset;
	baseRightLimit = x_offset + FRAMES[0].cols;
	rightLimit = scene_cornersRight[0].x;
	fourLimit = scene_cornersFour[0].x;
	Mat croppedImage;

	//for cropping final result PLEASE REDO
	cv::Point topLeft, topRight, bottomLeft, bottomRight;
	int bottomLowerHeight, rightSmallerWidth, croppedWidth, croppedHeight;
	topLeft.y = scene_cornersFive[0].y;
	topLeft.x = scene_cornersFive[0].x;
	topRight.y = scene_cornersFour[1].y;
	topRight.x = scene_cornersFour[1].x;
	bottomLeft.y = scene_cornersFive[2].y;
	bottomLeft.x = scene_cornersFive[2].x;
	bottomRight.y = scene_cornersFour[3].y;
	bottomRight.x = scene_cornersFour[3].x;

	if (topLeft.y < 0)
		topLeft.y = 0;
	if (topLeft.x < 0)
		topLeft.x = 0;
	if (topRight.y < 0)
		topRight.y = 0;
	if (topRight.x > result.cols)
		topRight.x = result.cols;
	if (bottomRight.y > result.rows)
		bottomRight.y = result.rows;
	if (bottomRight.x > result.cols)
		bottomRight.x = result.cols;
	if (bottomLeft.y > result.rows)
		bottomLeft.y = result.rows;
	if (bottomLeft.x < 0)
		bottomLeft.x = 0;

	(bottomLeft.y < bottomRight.y) ? bottomLowerHeight = bottomLeft.y : bottomLowerHeight = bottomRight.y;
	(topRight.x < bottomRight.x) ? rightSmallerWidth = topRight.x : rightSmallerWidth = bottomRight.x;
	(topLeft.x < bottomLeft.x) ? topLeft.x = bottomLeft.x : topLeft.x = topLeft.x;
	(topLeft.y < topRight.y) ? topLeft.y = topRight.y : topLeft.y = topLeft.y;
	croppedWidth = rightSmallerWidth - topLeft.x;
	croppedHeight = bottomLowerHeight - topLeft.y;

	//Timer Info
	int _frequency = getTickFrequency();
	float _secsPerCycle = (float)1 / _frequency;
	int frameNo = 0;
	float _totalSPF = 0;

	//Get GPU ready
	//cv::gpu::setDevice(0);
	//cv::Mat tmp(FRAMES[0]);
	//capL.read(tmp);
	//tmp = FRAMES[0];
	//cv::gpu::GpuMat templ_d(tmp); // Warm up the cores

	//Initialize needed variables for GPU
	cv::Mat outLeftFrame, outRightFrame, outBaseFrame, outFourFrame, outFiveFrame, outSixFrame;
	RESULTS[0] = outLeftFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[1] = outRightFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[2] = outBaseFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[3] = outFourFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[4] = outFiveFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	//outSixFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);

	cv::gpu::Stream streamL, streamR, streamB, stream4, stream5, stream6;

	//Start processing
	while (1)
	{
		frameNo++;
		int _startWhileLoop = (int)getTickCount();

		CO->CO_captureFrames(FRAMES);
		//imshow("base Frame", baseFrame);
		IO->IO_cvtColor(FRAMES, CV_RGB2GRAY);
		IO->IO_resize(FRAMES, cv::Size(frameWidth, frameHeight));
		IO->IO_transpose(FRAMES);
		IO->IO_flip(FRAMES, 1);
		IO->IO_undistort(FRAMES, INTRINSICCOEFFS, DISTORTIONCOEFFS);
		IO->IO_rectilinearProject(FRAMES, 0, FOCAL);
		GO->GO_uploadStream(FRAMES);
		GO->GO_perspectiveTransform(EXTRINSICCOEFFS, resultHeight + 600, resultWidth);
		GO->GO_downloadStream(RESULTS);

		if (!useGrayScale) {

			/// JH: Added RGB support using cv::Vec3b when grayScale option is disabled
			for (int j = 0; j < result.rows; ++j)
				for (int i = 0; i < result.cols; ++i)
				{
					//cout << "blending" << endl;
					/**
					cv::Vec3b cL(0, 0, 0);
					cv::Vec3b cB(0, 0, 0);
					cv::Vec3b cR(0, 0, 0);
					cv::Vec3b cLB(0, 0, 0);
					cv::Vec3b cBR(0, 0, 0);
					cv::Vec3b color(0, 0, 0);
					*/
					float blendA = 0.8;
					cv::Vec3b cL;
					cv::Vec3b cB;
					cv::Vec3b cR;
					cv::Vec3b c4;
					cv::Vec3b c5;
					cv::Vec3b c6;
					cv::Vec3b cLB;
					cv::Vec3b cBR;
					cv::Vec3b cR4;
					cv::Vec3b c5L;
					cv::Vec3b c65;
					cv::Vec3b color;

					float coeff = 0.4;
					int blendValue;
					bool cL_0 = false;
					bool cB_0 = false;
					bool cR_0 = false;
					bool c4_0 = false;
					bool c5_0 = false;
					bool c6_0 = false;


					//color = resultB.at<uchar>(j, i) + resultL.at<uchar>(j, i) + resultR.at<uchar>(j, i) + result4.at<uchar>(j, i) + result5.at<uchar>(j, i);


					// Assign flags
					/*
					if (j < result.rows && i < sixLimit){
					c6_0 = true;
					c6 = outSixFrame.at<uchar>(j, i);
					}
					*/
					if (j < result.rows && i < fifthLimit){
						c5_0 = true;
						c5 = outFiveFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i < leftLimit && i > fifthLimit){
						cL_0 = true;
						cL = outLeftFrame.at<cv::Vec3b>(j, i);
					}
					if (j < baseFrame.rows && i>baseLeftLimit && i < baseRightLimit) {
						//cout << "cB is true" << endl;
						cB_0 = true;
						cB = outBaseFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i>rightLimit && i < fourLimit) {
						cR_0 = true;
						cR = outRightFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i> fourLimit) {
						c4_0 = true;
						c4 = outFourFrame.at<cv::Vec3b>(j, i);
					}


					// Activate color based on flags
					if (c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// Use combination of five + left
						color = ((1 - blendA)*cL + blendA*c5);
					}
					else if (!c5_0 && cL_0 && cB_0 && !cR_0)
					{
						// Use combination of base + left
						color = ((1 - blendA)*cL + blendA*cB);
					}
					else if (!c5_0 && !cL_0 && cB_0 && cR_0 && !c4_0)
					{
						// Use combination of base + right
						color = ((1 - blendA)*cB + blendA*cR);
					}
					else if (!c5_0 && !cL_0 && !cB_0 && cR_0 && c4_0)
					{
						// Use combination of four + right
						color = ((1 - blendA)*cR + blendA*c4);
					}
					/*
					else if (c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// Use combination of six + five
					color = ((1 - blendA)*c5 + blendA*c6);
					}
					*/
					else if (!c6_0 && !c5_0 && !cL_0 && cB_0 && !cR_0 && !c4_0)
					{
						// In base's frame
						color = cB;
					}
					else if (!c6_0 && !c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// In left's frame
						color = cL;
					}

					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && cR_0 && !c4_0)
					{
						// In right frame
						color = cR;
					}
					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && c4_0)
					{
						// In fourth frame
						color = c4;
					}
					else if (!c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
						// In fifth frame
						color = c5;
					}
					/*
					else if (c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// In sixth frame
					color = c6;
					}
					*/
					//result.at<cv::Vec3b>(j, i) = color;
					result.at<cv::Vec3b>(j, i) = color;
				}
		}
		else {
			for (int j = 0; j < result.rows; ++j)
				for (int i = 0; i < result.cols; ++i)
				{
					
					float blendA = 0.8;
					uchar cL;
					uchar cB;
					uchar cR;
					uchar c4;
					uchar c5;
					uchar c6;
					uchar cLB;
					uchar cBR;
					uchar cR4;
					uchar c5L;
					uchar c65;
					uchar color;

					float coeff = 0.4;
					int blendValue;
					bool cL_0 = false;
					bool cB_0 = false;
					bool cR_0 = false;
					bool c4_0 = false;
					bool c5_0 = false;
					bool c6_0 = false;


					//color = resultB.at<uchar>(j, i) + resultL.at<uchar>(j, i) + resultR.at<uchar>(j, i) + result4.at<uchar>(j, i) + result5.at<uchar>(j, i);


					// Assign flags
					/*
					if (j < result.rows && i < sixLimit){
					c6_0 = true;
					c6 = outSixFrame.at<uchar>(j, i);
					}
					*/
					if (j < result.rows && i < fifthLimit){
						c5_0 = true;
						c5 = RESULTS[4].at<uchar>(j, i);
					}
					if (j < result.rows && i < leftLimit && i > fifthLimit){
						cL_0 = true;
						cL = RESULTS[1].at<uchar>(j, i);
					}
					if (j < result.rows && i > baseLeftLimit && i < baseRightLimit) {
						//cout << "cB is true" << endl;
						cB_0 = true;
						cB = RESULTS[0].at<uchar>(j, i);
					}
					if (j < result.rows && i>rightLimit && i < fourLimit) {
						cR_0 = true;
						cR = RESULTS[2].at<uchar>(j, i);
					}
					if (j < result.rows && i> fourLimit) {
						c4_0 = true;
						c4 = RESULTS[3].at<uchar>(j, i);
					}



					// Activate color based on flags
					if (c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// Use combination of five + left
						color = ((1 - blendA)*cL + blendA*c5);
					}
					else if (!c5_0 && cL_0 && cB_0 && !cR_0)
					{
						// Use combination of base + left
						color = ((1 - blendA)*cL + blendA*cB);
					}
					else if (!c5_0 && !cL_0 && cB_0 && cR_0 && !c4_0)
					{
						// Use combination of base + right
						color = ((1 - blendA)*cB + blendA*cR);
					}
					else if (!c5_0 && !cL_0 && !cB_0 && cR_0 && c4_0)
					{
						// Use combination of four + right
						color = ((1 - blendA)*cR + blendA*c4);
					}
					/*
					else if (c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// Use combination of six + five
					color = ((1 - blendA)*c5 + blendA*c6);
					}
					*/
					else if (!c6_0 && !c5_0 && !cL_0 && cB_0 && !cR_0 && !c4_0)
					{
						// In base's frame
						color = cB;
					}
					else if (!c6_0 && !c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// In left's frame
						color = cL;
					}

					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && cR_0 && !c4_0)
					{
						// In right frame
						color = cR;
					}
					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && c4_0)
					{
						// In fourth frame
						color = c4;
					}
					else if (!c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
						// In fifth frame
						color = c5;
					}
					/*
					else if (c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// In sixth frame
					color = c6;
					}
					*/
					//result.at<cv::Vec3b>(j, i) = color;
					result.at<uchar>(j, i) = color;

				}
		}

		croppedImage = result(Rect(topLeft.x, topLeft.y, croppedWidth, croppedHeight));

		if (croppedImage.channels() == 3) {
			cv::cvtColor(croppedImage, croppedImage, CV_RGB2BGR);
		}

		imshow("cropped Result", croppedImage);

		//Latency Calculations
		int _endWhileLoop = (int)getTickCount();
		int _WhileLoopDiff = _endWhileLoop - _startWhileLoop;
		float _secsForWhileLoop = (float)(_secsPerCycle * _WhileLoopDiff);
		cout << "secs for Frame " << frameNo << " is " << _secsForWhileLoop << endl;
		_totalSPF = _totalSPF + _secsForWhileLoop;

		if ((frameNo % 30) == 0)
		{
			float _aveSPF = (float)_totalSPF / 30.0;
			cout << "Average Seconds-Per-Frame for past 30 frames is: " << _aveSPF << endl;
			_totalSPF = 0;
		}
		if (waitKey(30) == 110)
		{
			cout << "Saved Picture" << endl;
			cv::imwrite("Result_2.jpg", croppedImage);
		}
		if (waitKey(30) == 27)
			break;
	}
	return 1;


}

int stitchLiveWOGPU() {
	cout << "Stitching without GPU... \n" << endl;

	std::vector<int> cameraPorts(NO_OF_CAMS);
	cameraPorts[0] = BASE_CAM;
	cameraPorts[1] = LEFT_CAM;
	cameraPorts[2] = RIGHT_CAM;
	cameraPorts[3] = FOUR_CAM;
	cameraPorts[4] = FIFTH_CAM;
	CameraOps *CO = new CameraOps(cameraPorts);

	cv::VideoWriter outputVideo;
	cv::Mat result, leftFrame, baseFrame, rightFrame, fourFrame, fiveFrame, sixFrame;
	cv::Mat undistortedLeftFrame, undistortedBaseFrame, undistortedRightFrame, undistortedFourFrame, undistortedFiveFrame, undistortedSixFrame;

	//gpu::GpuMat resultL, resultB, resultR, resultMask, result4, result5, result6;
	cv::Mat resultL, resultB, resultR, resultMask, result4, result5, result6;

	std::vector<cv::Point2f> scene_cornersLeft, scene_cornersRight, scene_cornersBase, scene_cornersFour, scene_cornersFive, scene_cornersSix, scene_corners;
	CO->CO_setProp(CV_CAP_PROP_FRAME_WIDTH, 1920);
	CO->CO_setProp(CV_CAP_PROP_FRAME_HEIGHT, 1080);

	double Brightness;
	double Contrast;
	double Saturation;
	double Gain;

	Brightness = CO->CO_getProp(CV_CAP_PROP_BRIGHTNESS, 0);
	Contrast = CO->CO_getProp(CV_CAP_PROP_CONTRAST, 0);
	Saturation = CO->CO_getProp(CV_CAP_PROP_SATURATION, 0);
	Gain = CO->CO_getProp(CV_CAP_PROP_GAIN, 0);

	cout << "Brightness: " << Brightness;
	cout << "Contrast: " << Contrast;
	cout << "Saturation: " << Saturation;
	cout << "Gain: " << Gain;

	CO->CO_setProp(CV_CAP_PROP_BRIGHTNESS, Brightness);
	CO->CO_setProp(CV_CAP_PROP_CONTRAST, Contrast);
	CO->CO_setProp(CV_CAP_PROP_SATURATION, Saturation);
	CO->CO_setProp(CV_CAP_PROP_GAIN, Gain);

	int frameWidth = CO->CO_getProp(CV_CAP_PROP_FRAME_WIDTH,0)*0.25;
	int frameHeight = CO->CO_getProp(CV_CAP_PROP_FRAME_HEIGHT, 0)*0.25;
	int resultWidth = frameHeight * 2;
	int resultHeight = frameWidth + 100;
	bool record = false;
	RESULTS[0] = resultB = cv::Mat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[1] = resultL = cv::Mat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[2] = resultR = cv::Mat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[3] = result4 = cv::Mat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[4] = result5 = cv::Mat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);

	result = Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);

	ImageOps *IO = new ImageOps();

	// Move Scene to the right by 100
	int x_offset = 500;
	float y_offset = 0.0;
	float z_offset = 100.0;
	float transdata[] = { 1.0, 0.0, x_offset, 0.0, 1.0, y_offset, 0.0, 0.0, 1.0 };
	cv::Mat trans(3, 3, CV_32FC1, transdata);
	EXTRINSICCOEFFS[0] = trans;

	Mat HR_m = HR.clone();
	Mat HL_m = HL.clone();
	HR = trans * HR;
	HL = trans * HL;
	H4 = trans * HR_m * H4;
	H5 = trans * HL_m * H5;
	//H6 = trans * HL_m * H5 * H6;

	cout << "finished getting matrix" << endl;

	CO->CO_captureFrames(FRAMES);

	if (FRAMES[0].cols == 0) {
		cout << "Error reading file " << endl;
		return -1;
	}

	IO->IO_resize(FRAMES, cv::Size(frameWidth, frameHeight));
	IO->IO_transpose(FRAMES);
	IO->IO_flip(FRAMES, 1);
	IO->IO_undistort(FRAMES, INTRINSICCOEFFS, DISTORTIONCOEFFS);

	//MOVE TO PREPROCESSING
	scene_corners.clear();
	scene_cornersLeft.push_back(Point2f(0.0, 0.0));
	scene_cornersLeft.push_back(Point2f(FRAMES[1].cols, 0.0));
	scene_cornersLeft.push_back(Point2f(0.0, FRAMES[1].rows));
	scene_cornersLeft.push_back(Point2f(FRAMES[1].cols, FRAMES[1].rows));
	scene_cornersRight.push_back(Point2f(0.0, 0.0));
	scene_cornersRight.push_back(Point2f(FRAMES[2].cols, 0.0));
	scene_cornersRight.push_back(Point2f(0.0, FRAMES[2].rows));
	scene_cornersRight.push_back(Point2f(FRAMES[2].cols, FRAMES[2].rows));
	scene_cornersBase.push_back(Point2f(0.0, 0.0));
	scene_cornersBase.push_back(Point2f(FRAMES[0].cols, 0.0));
	scene_cornersBase.push_back(Point2f(0.0, FRAMES[0].rows));
	scene_cornersBase.push_back(Point2f(FRAMES[0].cols, FRAMES[0].rows));
	scene_cornersFour.push_back(Point2f(0.0, 0.0));
	scene_cornersFour.push_back(Point2f(FRAMES[3].cols, 0.0));
	scene_cornersFour.push_back(Point2f(0.0, FRAMES[3].rows));
	scene_cornersFour.push_back(Point2f(FRAMES[3].cols, FRAMES[3].rows));
	scene_cornersFive.push_back(Point2f(0.0, 0.0));
	scene_cornersFive.push_back(Point2f(FRAMES[4].cols, 0.0));
	scene_cornersFive.push_back(Point2f(0.0, FRAMES[4].rows));
	scene_cornersFive.push_back(Point2f(FRAMES[4].cols, FRAMES[4].rows));

	//MOVE TO PREPROCESSING
	perspectiveTransform(scene_cornersBase, scene_cornersBase, trans);
	perspectiveTransform(scene_cornersLeft, scene_cornersLeft, HL);
	perspectiveTransform(scene_cornersRight, scene_cornersRight, HR);
	perspectiveTransform(scene_cornersFour, scene_cornersFour, H4);
	perspectiveTransform(scene_cornersFive, scene_cornersFive, H5);

	//Store useful information for Image limits
	int leftLimit, baseLeftLimit, baseRightLimit, rightLimit, fourLimit, fifthLimit, sixLimit;

	//MOVE TO PREPROCESSING
	fifthLimit = scene_cornersFive[1].x;
	//sixLimit = scene_cornersSix[1].x;
	leftLimit = scene_cornersLeft[1].x;
	baseLeftLimit = x_offset;
	baseRightLimit = x_offset + FRAMES[0].cols;
	rightLimit = scene_cornersRight[0].x;
	fourLimit = scene_cornersFour[0].x;
	Mat croppedImage;

	//for cropping final result PLEASE REDO
	//MOVE TO PREPROCESSING
	cv::Point topLeft, topRight, bottomLeft, bottomRight;
	int bottomLowerHeight, rightSmallerWidth, croppedWidth, croppedHeight;
	topLeft.y = scene_cornersFive[0].y;
	topLeft.x = scene_cornersFive[0].x;
	topRight.y = scene_cornersFour[1].y;
	topRight.x = scene_cornersFour[1].x;
	bottomLeft.y = scene_cornersFive[2].y;
	bottomLeft.x = scene_cornersFive[2].x;
	bottomRight.y = scene_cornersFour[3].y;
	bottomRight.x = scene_cornersFour[3].x;

	//MOVE TO PREPROCESSING
	if (topLeft.y < 0)
		topLeft.y = 0;
	if (topLeft.x < 0)
		topLeft.x = 0;
	if (topRight.y < 0)
		topRight.y = 0;
	if (topRight.x > result.cols)
		topRight.x = result.cols;
	if (bottomRight.y > result.rows)
		bottomRight.y = result.rows;
	if (bottomRight.x > result.cols)
		bottomRight.x = result.cols;
	if (bottomLeft.y > result.rows)
		bottomLeft.y = result.rows;
	if (bottomLeft.x < 0)
		bottomLeft.x = 0;

	//MOVE TO PREPROCESSING
	(bottomLeft.y < bottomRight.y) ? bottomLowerHeight = bottomLeft.y : bottomLowerHeight = bottomRight.y;
	(topRight.x < bottomRight.x) ? rightSmallerWidth = topRight.x : rightSmallerWidth = bottomRight.x;
	(topLeft.x < bottomLeft.x) ? topLeft.x = bottomLeft.x : topLeft.x = topLeft.x;
	(topLeft.y < topRight.y) ? topLeft.y = topRight.y : topLeft.y = topLeft.y;
	croppedWidth = rightSmallerWidth - topLeft.x;
	croppedHeight = bottomLowerHeight - topLeft.y;

	//Timer Info
	int _frequency = getTickFrequency();
	float _secsPerCycle = (float)1 / _frequency;
	int frameNo = 0;
	float _totalSPF = 0;

	//Get GPU ready
	//cv::gpu::setDevice(0);
	//cv::Mat tmp;
	
	//cv::gpu::GpuMat templ_d(tmp); // Warm up the cores

	//Initialize needed variables for GPU
	cv::gpu::GpuMat imageBSrc, imageBDst, imageRSrc, imageRDst, imageLSrc, image4Src, image4Dst, image5Src, image5Dst, imageLDst, image6Dst, image6Src;
	cv::Mat outLeftFrame, outRightFrame, outBaseFrame, outFourFrame, outFiveFrame, outSixFrame;
	outLeftFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outRightFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outBaseFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outFourFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outFiveFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outSixFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);

	cv::gpu::Stream streamL, streamR, streamB, stream4, stream5, stream6;

	//Start processing
	while (1)
	{
		frameNo++;
		int _startWhileLoop = (int)getTickCount();

		CO->CO_captureFrames(FRAMES);
		//imshow("base Frame", baseFrame);
		if (useGrayScale) {
			IO->IO_cvtColor(FRAMES, CV_RGB2GRAY);
		}
		else {
			cv::Mat rrr;
			cvtColor(rightFrame, rrr, CV_RGB2BGR);
			rightFrame = rrr.clone();
			cvtColor(baseFrame, rrr, CV_RGB2BGR);
			baseFrame == rrr.clone();
			//imshow("test Frame", baseFrame);
			cvtColor(leftFrame, rrr, CV_RGB2BGR);
			leftFrame = rrr.clone();
			cvtColor(fourFrame, rrr, CV_RGB2BGR);
			fourFrame = rrr.clone();
			cvtColor(fiveFrame, rrr, CV_RGB2BGR);
			fiveFrame = rrr.clone();
		}

		IO->IO_resize(FRAMES, cv::Size(frameWidth, frameHeight));
		IO->IO_transpose(FRAMES);
		IO->IO_flip(FRAMES, 1);
		IO->IO_undistort(FRAMES, INTRINSICCOEFFS, DISTORTIONCOEFFS);
		IO->IO_rectilinearProject(FRAMES, 0, FOCAL);
		IO->IO_warpPerspective(FRAMES, RESULTS, EXTRINSICCOEFFS, cv::Size(resultHeight + 600, resultWidth));

		if (!useGrayScale) {

			/// JH: Added RGB support using cv::Vec3b when grayScale option is disabled
			for (int j = 0; j < result.rows; ++j)
				for (int i = 0; i < result.cols; ++i)
				{

					float blendA = 0.8;
					cv::Vec3b cL;
					cv::Vec3b cB;
					cv::Vec3b cR;
					cv::Vec3b c4;
					cv::Vec3b c5;
					cv::Vec3b c6;
					cv::Vec3b cLB;
					cv::Vec3b cBR;
					cv::Vec3b cR4;
					cv::Vec3b c5L;
					cv::Vec3b c65;
					cv::Vec3b color;

					float coeff = 0.4;
					int blendValue;
					bool cL_0 = false;
					bool cB_0 = false;
					bool cR_0 = false;
					bool c4_0 = false;
					bool c5_0 = false;
					bool c6_0 = false;


					//color = resultB.at<uchar>(j, i) + resultL.at<uchar>(j, i) + resultR.at<uchar>(j, i) + result4.at<uchar>(j, i) + result5.at<uchar>(j, i);


					// Assign flags
					/*
					if (j < result.rows && i < sixLimit){
					c6_0 = true;
					c6 = outSixFrame.at<uchar>(j, i);
					}
					*/
					if (j < result.rows && i < fifthLimit){
						c5_0 = true;
						c5 = outFiveFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i < leftLimit && i > fifthLimit){
						cL_0 = true;
						cL = outLeftFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i>baseLeftLimit && i < baseRightLimit) {
						cout << "cB is true" << endl;
						cB_0 = true;
						cB = outBaseFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i>rightLimit && i < fourLimit) {
						cR_0 = true;
						cR = outRightFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i> fourLimit) {
						c4_0 = true;
						c4 = outFourFrame.at<cv::Vec3b>(j, i);
					}


					// Activate color based on flags
					if (c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// Use combination of five + left
						color = ((1 - blendA)*cL + blendA*c5);
					}
					else if (!c5_0 && cL_0 && cB_0 && !cR_0)
					{
						// Use combination of base + left
						color = ((1 - blendA)*cL + blendA*cB);
					}
					else if (!c5_0 && !cL_0 && cB_0 && cR_0 && !c4_0)
					{
						// Use combination of base + right
						color = ((1 - blendA)*cB + blendA*cR);
					}
					else if (!c5_0 && !cL_0 && !cB_0 && cR_0 && c4_0)
					{
						// Use combination of four + right
						color = ((1 - blendA)*cR + blendA*c4);
					}
					/*
					else if (c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// Use combination of six + five
					color = ((1 - blendA)*c5 + blendA*c6);
					}
					*/
					else if (!c6_0 && !c5_0 && !cL_0 && cB_0 && !cR_0 && !c4_0)
					{
						// In base's frame
						color = cB;
					}
					else if (!c6_0 && !c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// In left's frame
						color = cL;
					}

					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && cR_0 && !c4_0)
					{
						// In right frame
						color = cR;
					}
					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && c4_0)
					{
						// In fourth frame
						color = c4;
					}
					else if (!c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
						// In fifth frame
						color = c5;
					}
					/*
					else if (c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// In sixth frame
					color = c6;
					}
					*/
					//result.at<cv::Vec3b>(j, i) = color;
					result.at<cv::Vec3b>(j, i) = color;
				}
		}
		else {
			for (int j = 0; j < result.rows; ++j)
				for (int i = 0; i < result.cols; ++i)
				{
					//cout << "blending" << endl;
					/**
					cv::Vec3b cL(0, 0, 0);
					cv::Vec3b cB(0, 0, 0);
					cv::Vec3b cR(0, 0, 0);
					cv::Vec3b cLB(0, 0, 0);
					cv::Vec3b cBR(0, 0, 0);
					cv::Vec3b color(0, 0, 0);
					*/
					float blendA = 0.8;
					uchar cL;
					uchar cB;
					uchar cR;
					uchar c4;
					uchar c5;
					uchar c6;
					uchar cLB;
					uchar cBR;
					uchar cR4;
					uchar c5L;
					uchar c65;
					uchar color;

					float coeff = 0.4;
					int blendValue;
					bool cL_0 = false;
					bool cB_0 = false;
					bool cR_0 = false;
					bool c4_0 = false;
					bool c5_0 = false;
					bool c6_0 = false;


					//color = resultB.at<uchar>(j, i) + resultL.at<uchar>(j, i) + resultR.at<uchar>(j, i) + result4.at<uchar>(j, i) + result5.at<uchar>(j, i);


					// Assign flags
					/*
					if (j < result.rows && i < sixLimit){
					c6_0 = true;
					c6 = outSixFrame.at<uchar>(j, i);
					}
					*/
					if (j < result.rows && i < fifthLimit){
						c5_0 = true;
						c5 = RESULTS[4].at<uchar>(j, i);
					}
					if (j < result.rows && i < leftLimit && i > fifthLimit){
						cL_0 = true;
						cL = RESULTS[1].at<uchar>(j, i);
					}
					if (j < result.rows && i > baseLeftLimit && i < baseRightLimit) {
						//cout << "cB is true" << endl;
						cB_0 = true;
						cB = RESULTS[0].at<uchar>(j, i);
					}
					if (j < result.rows && i>rightLimit && i < fourLimit) {
						cR_0 = true;
						cR = RESULTS[2].at<uchar>(j, i);
					}
					if (j < result.rows && i> fourLimit) {
						c4_0 = true;
						c4 = RESULTS[3].at<uchar>(j, i);
					}



					// Activate color based on flags
					if (c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// Use combination of five + left
						color = ((1 - blendA)*cL + blendA*c5);
					}
					else if (!c5_0 && cL_0 && cB_0 && !cR_0)
					{
						// Use combination of base + left
						color = ((1 - blendA)*cL + blendA*cB);
					}
					else if (!c5_0 && !cL_0 && cB_0 && cR_0 && !c4_0)
					{
						// Use combination of base + right
						color = ((1 - blendA)*cB + blendA*cR);
					}
					else if (!c5_0 && !cL_0 && !cB_0 && cR_0 && c4_0)
					{
						// Use combination of four + right
						color = ((1 - blendA)*cR + blendA*c4);
					}
					/*
					else if (c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// Use combination of six + five
					color = ((1 - blendA)*c5 + blendA*c6);
					}
					*/
					else if (!c6_0 && !c5_0 && !cL_0 && cB_0 && !cR_0 && !c4_0)
					{
						// In base's frame
						color = cB;
					}
					else if (!c6_0 && !c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// In left's frame
						color = cL;
					}

					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && cR_0 && !c4_0)
					{
						// In right frame
						color = cR;
					}
					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && c4_0)
					{
						// In fourth frame
						color = c4;
					}
					else if (!c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
						// In fifth frame
						color = c5;
					}
					/*
					else if (c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// In sixth frame
					color = c6;
					}
					*/
					//result.at<cv::Vec3b>(j, i) = color;
					result.at<uchar>(j, i) = color;

				}
		}

		croppedImage = result(Rect(topLeft.x, topLeft.y, croppedWidth, croppedHeight));
		if (croppedImage.channels() == 3) {
			cv::cvtColor(croppedImage, croppedImage, CV_RGB2BGR);
		}

		imshow("cropped Result", croppedImage);

		//Latency Calculations
		int _endWhileLoop = (int)getTickCount();
		int _WhileLoopDiff = _endWhileLoop - _startWhileLoop;
		float _secsForWhileLoop = (float)(_secsPerCycle * _WhileLoopDiff);
		cout << "secs for Frame " << frameNo << " is " << _secsForWhileLoop << endl;
		_totalSPF = _totalSPF + _secsForWhileLoop;

		if ((frameNo % 30) == 0)
		{
			float _aveSPF = (float)_totalSPF / 30.0;
			cout << "Average Seconds-Per-Frame for past 30 frames is: " << _aveSPF << endl;
			_totalSPF = 0;
		}
		if (waitKey(30) == 110)
		{
			cout << "Saved Picture" << endl;
			cv::imwrite("Result_2.jpg", croppedImage);
		}
		if (waitKey(30) == 27)
			break;
	}
}

int recordSimple() {

	// Create blank image for instructions
	cv::Mat directions_screen = cv::Mat(400, 300, CV_8UC3);
	directions_screen.setTo(cv::Scalar(130, 120, 180));
	putText(directions_screen, "Press on SPACE to Record!", cvPoint(30, 30),
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);

	imshow("Directions Screen", directions_screen);
	
	// Get camera ready
	cv::VideoCapture capB(BASE_CAM);
	cv::VideoCapture capR(RIGHT_CAM);
	cv::VideoCapture capL(LEFT_CAM);
	cv::VideoCapture cap4(FOUR_CAM);
	cv::VideoCapture cap5(FIFTH_CAM);
	cv::VideoWriter outputVideoB, outputVideoL, outputVideoR, outputVideo4, outputVideo5;
	capL.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	capL.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	capR.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	capR.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	capB.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	capB.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	cap4.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	cap4.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	cap5.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	cap5.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	int frameWidth = capL.get(CV_CAP_PROP_FRAME_WIDTH)*0.25;
	int frameHeight = capL.get(CV_CAP_PROP_FRAME_HEIGHT)*0.25;
	cv::Mat baseFrame, leftFrame, rightFrame, fourFrame, fiveFrame;
	std::vector<cv::Mat> FRAMES(NO_OF_CAMS);

	ImageOps *IO = new ImageOps();

	//uni settings 
	double Brightness;
	double Contrast;
	double Saturation;
	double Gain;

	Brightness = capB.get(CV_CAP_PROP_BRIGHTNESS);
	Contrast = capB.get(CV_CAP_PROP_CONTRAST);
	Saturation = capB.get(CV_CAP_PROP_SATURATION);
	Gain = capB.get(CV_CAP_PROP_GAIN);

	cout << "Brightness: " << Brightness;
	cout << "Contrast: " << Contrast;
	cout << "Saturation: " << Saturation;
	cout << "Gain: " << Gain;

	capB.set(CV_CAP_PROP_BRIGHTNESS, Brightness);
	capL.set(CV_CAP_PROP_BRIGHTNESS, Brightness);
	capR.set(CV_CAP_PROP_BRIGHTNESS, Brightness);
	cap4.set(CV_CAP_PROP_BRIGHTNESS, Brightness);
	cap5.set(CV_CAP_PROP_BRIGHTNESS, Brightness);

	capB.set(CV_CAP_PROP_CONTRAST, Contrast);
	capL.set(CV_CAP_PROP_CONTRAST, Contrast);
	capR.set(CV_CAP_PROP_CONTRAST, Contrast);
	cap4.set(CV_CAP_PROP_CONTRAST, Contrast);
	cap5.set(CV_CAP_PROP_CONTRAST, Contrast);

	capB.set(CV_CAP_PROP_SATURATION, Saturation);
	capL.set(CV_CAP_PROP_SATURATION, Saturation);
	capR.set(CV_CAP_PROP_SATURATION, Saturation);
	cap4.set(CV_CAP_PROP_SATURATION, Saturation);
	cap5.set(CV_CAP_PROP_SATURATION, Saturation);

	capB.set(CV_CAP_PROP_GAIN, Gain);
	capL.set(CV_CAP_PROP_GAIN, Gain);
	capR.set(CV_CAP_PROP_GAIN, Gain);
	cap4.set(CV_CAP_PROP_GAIN, Gain);
	cap5.set(CV_CAP_PROP_GAIN, Gain);

	outputVideoL.open(camLOutput, -1, 30, cv::Size(frameHeight, frameWidth), true);
	outputVideoR.open(camROutput, -1, 30, cv::Size(frameHeight, frameWidth), true);
	outputVideoB.open(camBOutput, -1, 30, cv::Size(frameHeight, frameWidth), true);
	outputVideo4.open(cam4Output, -1, 30, cv::Size(frameHeight, frameWidth), true);
	outputVideo5.open(cam5Output, -1, 30, cv::Size(frameHeight, frameWidth), true);

	while (1) {
		capB >> FRAMES[0];
		capL >> FRAMES[1];
		capR >> FRAMES[2];
		cap4 >> FRAMES[3];
		cap5 >> FRAMES[4];


		IO->IO_resize(FRAMES, cv::Size(frameWidth, frameHeight));
		IO->IO_transpose(FRAMES);
		IO->IO_flip(FRAMES, 1);

		outputVideoB << FRAMES[0];
		outputVideoL << FRAMES[1];
		outputVideoR << FRAMES[2];
		outputVideo4 << FRAMES[3];
		outputVideo5 <<	FRAMES[4];

		imshow("Base", FRAMES[0]);
		imshow("Left", FRAMES[1]);
		imshow("Right", FRAMES[2]);
		imshow("Four", FRAMES[3]);
		imshow("Five", FRAMES[4]);

		if (waitKey(30) == ' ')
			break;
	}

	return 1;
}

int record()
{
	// Create blank image for instructions
	cv::Mat directions_screen = cv::Mat(400, 300, CV_8UC3);
	directions_screen.setTo(cv::Scalar(130, 120, 180));
	putText(directions_screen, "Press on SPACE to Record!", cvPoint(30, 30),
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);

	imshow("Directions Screen", directions_screen);
	while (1)
	{
		if (waitKey(30) == ' ')
		{
			directions_screen.setTo(cv::Scalar(255, 255, 180));
			putText(directions_screen, "Recording! Press SPACE to stop.", cvPoint(30, 30),
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);

			imshow("Directions Screen", directions_screen);

			cv::VideoCapture capB(BASE_CAM);
			cv::VideoCapture capR(RIGHT_CAM);
			cv::VideoCapture capL(LEFT_CAM);
			cv::VideoCapture cap4(FOUR_CAM);
			cv::VideoCapture cap5(FIFTH_CAM);
			cv::VideoWriter outputVideoB, outputVideoL, outputVideoR, outputVideo4, outputVideo5;
			capL.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
			capL.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
			capR.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
			capR.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
			capB.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
			capB.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
			cap4.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
			cap4.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
			cap5.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
			cap5.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
			int frameWidth = capL.get(CV_CAP_PROP_FRAME_WIDTH)*0.25;
			int frameHeight = capL.get(CV_CAP_PROP_FRAME_HEIGHT)*0.25;
			cv::Mat baseFrame, leftFrame, rightFrame, fourFrame, fiveFrame;
			//uni settings 
			double Brightness;
			double Contrast;
			double Saturation;
			double Gain;

			Brightness = capB.get(CV_CAP_PROP_BRIGHTNESS);
			Contrast = capB.get(CV_CAP_PROP_CONTRAST);
			Saturation = capB.get(CV_CAP_PROP_SATURATION);
			Gain = capB.get(CV_CAP_PROP_GAIN);

			cout << "Brightness: " << Brightness;
			cout << "Contrast: " << Contrast;
			cout << "Saturation: " << Saturation;
			cout << "Gain: " << Gain;

			capB.set(CV_CAP_PROP_BRIGHTNESS, Brightness);
			capL.set(CV_CAP_PROP_BRIGHTNESS, Brightness);
			capR.set(CV_CAP_PROP_BRIGHTNESS, Brightness);
			cap4.set(CV_CAP_PROP_BRIGHTNESS, Brightness);
			cap5.set(CV_CAP_PROP_BRIGHTNESS, Brightness);

			capB.set(CV_CAP_PROP_CONTRAST, Contrast);
			capL.set(CV_CAP_PROP_CONTRAST, Contrast);
			capR.set(CV_CAP_PROP_CONTRAST, Contrast);
			cap4.set(CV_CAP_PROP_CONTRAST, Contrast);
			cap5.set(CV_CAP_PROP_CONTRAST, Contrast);

			capB.set(CV_CAP_PROP_SATURATION, Saturation);
			capL.set(CV_CAP_PROP_SATURATION, Saturation);
			capR.set(CV_CAP_PROP_SATURATION, Saturation);
			cap4.set(CV_CAP_PROP_SATURATION, Saturation);
			cap5.set(CV_CAP_PROP_SATURATION, Saturation);

			capB.set(CV_CAP_PROP_GAIN, Gain);
			capL.set(CV_CAP_PROP_GAIN, Gain);
			capR.set(CV_CAP_PROP_GAIN, Gain);
			cap4.set(CV_CAP_PROP_GAIN, Gain);
			cap5.set(CV_CAP_PROP_GAIN, Gain);

			//Get GPU ready
			cv::gpu::setDevice(0);
			cv::Mat tmp;
			capL.read(tmp);
			cv::gpu::GpuMat templ_d(tmp); // Warm up the cores

			cv::gpu::GpuMat imageBSrc, imageBDst, imageRSrc, imageRDst, imageLSrc, image4Src, image4Dst, image5Src, image5Dst, imageLDst, image6Dst, image6Src;
			cv::Mat outLeftFrame, outRightFrame, outBaseFrame, outFourFrame, outFiveFrame, outSixFrame;
			outLeftFrame = cv::Mat(frameWidth, frameHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
			outRightFrame = cv::Mat(frameWidth, frameHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
			outBaseFrame = cv::Mat(frameWidth, frameHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
			outFourFrame = cv::Mat(frameWidth, frameHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
			outFiveFrame = cv::Mat(frameWidth, frameHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
			outSixFrame = cv::Mat(frameWidth, frameHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
			cv::gpu::Stream streamL, streamR, streamB, stream4, stream5, stream6;
			outputVideoL.open(camLOutput, -1, 30, cv::Size(frameHeight, frameWidth), true);
			outputVideoR.open(camROutput, -1, 30, cv::Size(frameHeight, frameWidth), true);
			outputVideoB.open(camBOutput, -1, 30, cv::Size(frameHeight, frameWidth), true);
			outputVideo4.open(cam4Output, -1, 30, cv::Size(frameHeight, frameWidth), true);
			outputVideo5.open(cam5Output, -1, 30, cv::Size(frameHeight, frameWidth), true);

			while (1)
			{
				capB >> baseFrame;
				capR >> rightFrame;
				capL >> leftFrame;
				cap4 >> fourFrame;
				cap5 >> fiveFrame;

				cv::circle(baseFrame, cv::Point(frameWidth / 2, frameHeight / 2), 5, cv::Scalar(0, 255, 0), 2);
				cv::circle(rightFrame, cv::Point(frameWidth / 2, frameHeight / 2), 5, cv::Scalar(0, 255, 0), 2);
				cv::circle(leftFrame, cv::Point(frameWidth / 2, frameHeight / 2), 5, cv::Scalar(0, 255, 0), 2);
				cv::circle(fourFrame, cv::Point(frameWidth / 2, frameHeight / 2), 5, cv::Scalar(0, 255, 0), 2);
				cv::circle(fiveFrame, cv::Point(frameWidth / 2, frameHeight / 2), 5, cv::Scalar(0, 255, 0), 2);



				//Upload back to GPU

				if (useGrayScale) {
					cv::cvtColor(leftFrame, leftFrame, CV_RGB2GRAY);

					cv::cvtColor(rightFrame, rightFrame, CV_RGB2GRAY);
					cv::cvtColor(baseFrame, baseFrame, CV_RGB2GRAY);
					cv::cvtColor(fourFrame, fourFrame, CV_RGB2GRAY);
					cv::cvtColor(fiveFrame, fiveFrame, CV_RGB2GRAY);

					streamL.enqueueUpload(leftFrame, imageLSrc);
					streamR.enqueueUpload(rightFrame, imageRSrc);
					streamB.enqueueUpload(baseFrame, imageBSrc);
					stream4.enqueueUpload(fourFrame, image4Src);
					stream5.enqueueUpload(fiveFrame, image5Src);
					std::cout << useGrayScale << " UseGrayScale enabled" << std::endl;
					/*	gpu::cvtColor(imageRSrc, imageRDst, CV_RGB2GRAY, streamR);
					gpu::cvtColor(imageBSrc, imageBDst, CV_RGB2GRAY, streamB);
					gpu::cvtColor(imageLSrc, imageLDst, CV_RGB2GRAY, streamL);
					gpu::cvtColor(image4Src, image4Dst, CV_RGB2GRAY, stream4);
					gpu::cvtColor(image5Src, image5Dst, CV_RGB2GRAY, stream5);*/
				}
				else {
					imageRDst = imageRSrc;
					imageBDst = imageBSrc;
					imageLDst = imageLSrc;
					image4Dst = image4Src;
					image5Dst = image5Src;
				}

				//cvtColor(sixFrame, sixFrame, CV_RGB2GRAY);
				gpu::resize(imageRDst, imageRSrc, cv::Size(frameWidth, frameHeight), streamR);
				gpu::resize(imageBDst, imageBSrc, cv::Size(frameWidth, frameHeight), streamB);
				gpu::resize(imageLDst, imageLSrc, cv::Size(frameWidth, frameHeight), streamL);
				gpu::resize(image4Dst, image4Src, cv::Size(frameWidth, frameHeight), stream4);
				gpu::resize(image5Dst, image5Src, cv::Size(frameWidth, frameHeight), stream5);

				//resize(sixFrame, sixFrame, cv::Size(frameWidth, frameHeight));
				gpu::transpose(imageRSrc, imageRDst, streamR);
				gpu::transpose(imageBSrc, imageBDst, streamB);
				gpu::transpose(imageLSrc, imageLDst, streamL);
				gpu::transpose(image4Src, image4Dst, stream4);
				gpu::transpose(image5Src, image5Dst, stream5);

				//cv::transpose(sixFrame, sixFrame);
				gpu::flip(imageRDst, imageRSrc, 1, streamR);
				gpu::flip(imageBDst, imageBSrc, 1, streamB);
				gpu::flip(imageLDst, imageLSrc, 1, streamL);
				gpu::flip(image4Dst, image4Src, 1, stream4);
				gpu::flip(image5Dst, image5Src, 1, stream5);

				streamL.enqueueDownload(imageLSrc, outLeftFrame);
				streamR.enqueueDownload(imageRSrc, outRightFrame);
				streamB.enqueueDownload(imageBSrc, outBaseFrame);
				stream4.enqueueDownload(image4Src, outFourFrame);
				stream5.enqueueDownload(image5Src, outFiveFrame);

				outputVideoL << outLeftFrame;
				outputVideoR << outRightFrame;
				outputVideoB << outBaseFrame;
				outputVideo4 << outFourFrame;
				outputVideo5 << outFiveFrame;

				imshow("Base", outBaseFrame);
				imshow("Right", outRightFrame);
				imshow("Left", outLeftFrame);
				imshow("Four", outFourFrame);
				imshow("Five", outFiveFrame);

				if (waitKey(30) == ' ')
					break;
			}
			break;
		}

	}

	return 1;
}

int stitch()
{
	cv::VideoCapture capL(camLOutput), capB(camBOutput), capR(camROutput), cap4(cam4Output), cap5(cam5Output);
	cv::VideoWriter outputVideo;
	cv::Mat result, leftFrame, baseFrame, rightFrame, fourFrame, fiveFrame, sixFrame;
	cv::Mat undistortedLeftFrame, undistortedBaseFrame, undistortedRightFrame, undistortedFourFrame, undistortedFiveFrame, undistortedSixFrame;

	gpu::GpuMat resultL, resultB, resultR, resultMask, result4, result5, result6;
	std::vector<cv::Point2f> scene_cornersLeft, scene_cornersRight, scene_cornersBase, scene_cornersFour, scene_cornersFive, scene_cornersSix, scene_corners;
	int frameWidth = 1920 * 0.25;
	int frameHeight = 1080 * 0.25;
	int resultWidth = frameHeight * 2;
	int resultHeight = frameWidth + 100;
	bool record = false;
	resultL = gpu::GpuMat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	resultR = gpu::GpuMat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	resultB = gpu::GpuMat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	result4 = gpu::GpuMat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	result5 = gpu::GpuMat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	result6 = gpu::GpuMat(resultWidth, resultHeight, useGrayScale ? CV_8UC1 : CV_8UC3);
	result = Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);

	// Move Scene to the right by 100
	int x_offset = 500;
	float y_offset = 0.0;
	float z_offset = 100.0;
	float transdata[] = { 1.0, 0.0, x_offset, 0.0, 1.0, y_offset, 0.0, 0.0, 1.0 };
	cv::Mat trans(3, 3, CV_32FC1, transdata);
	cout << "HR: " << HR << endl;

	Mat HR_m = HR.clone();
	Mat HL_m = HL.clone();
	HR = trans * HR;
	HL = trans * HL;
	H4 = trans * HR_m * H4;
	H5 = trans * HL_m * H5;
	//H6 = trans * HL_m * H5 * H6;

	cout << "finished getting matrix" << endl;

	capL.read(leftFrame);
	capR.read(rightFrame);
	capB.read(baseFrame);
	cap4.read(fourFrame);
	cap5.read(fiveFrame);

	//cap6.read(sixFrame);


	if (fiveFrame.cols == 0) {
		cout << "Error reading file " << endl;
		return -1;
	}


	resize(leftFrame, leftFrame, cv::Size(frameWidth, frameHeight));
	resize(baseFrame, baseFrame, cv::Size(frameWidth, frameHeight));
	resize(rightFrame, rightFrame, cv::Size(frameWidth, frameHeight));
	resize(fourFrame, fourFrame, cv::Size(frameWidth, frameHeight));
	resize(fiveFrame, fiveFrame, cv::Size(frameWidth, frameHeight));
	//resize(sixFrame, sixFrame, cv::Size(frameWidth, frameHeight));

	/*
	cv::transpose(baseFrame, baseFrame);
	cv::transpose(rightFrame, rightFrame);
	cv::transpose(leftFrame, leftFrame);
	cv::transpose(fourFrame, fourFrame);
	cv::transpose(fiveFrame, fiveFrame);
	//cv::transpose(sixFrame, sixFrame);
	cv::flip(baseFrame, baseFrame, 1);
	cv::flip(rightFrame, rightFrame, 1);
	cv::flip(leftFrame, leftFrame, 1);
	cv::flip(fourFrame, fourFrame, 1);
	cv::flip(fiveFrame, fiveFrame, 1);
	//cv::flip(sixFrame, sixFrame, 1);
	undistort(leftFrame, undistortedLeftFrame, leftIntrinsic, leftDistCoeffs);
	undistort(baseFrame, undistortedBaseFrame, baseIntrinsic, baseDistCoeffs);
	undistort(rightFrame, undistortedRightFrame, rightIntrinsic, rightDistCoeffs);
	undistort(fourFrame, undistortedFourFrame, fourIntrinsic, fourDistCoeffs);
	undistort(fiveFrame, undistortedFiveFrame, fiveIntrinsic, fiveDistCoeffs);
	//undistort(sixFrame, undistortedSixFrame, sixIntrinsic, sixDistCoeffs);
	leftFrame = undistortedLeftFrame;
	baseFrame = undistortedBaseFrame;
	rightFrame = undistortedRightFrame;
	fourFrame = undistortedFourFrame;
	fiveFrame = undistortedFiveFrame;
	//sixFrame = undistortedSixFrame;
	*/
	// Use the Homography Matrix to warp the images
	scene_corners.clear();
	scene_cornersLeft.push_back(Point2f(0.0, 0.0));
	scene_cornersLeft.push_back(Point2f(leftFrame.cols, 0.0));
	scene_cornersLeft.push_back(Point2f(0.0, leftFrame.rows));
	scene_cornersLeft.push_back(Point2f(leftFrame.cols, leftFrame.rows));
	scene_cornersRight.push_back(Point2f(0.0, 0.0));
	scene_cornersRight.push_back(Point2f(rightFrame.cols, 0.0));
	scene_cornersRight.push_back(Point2f(0.0, rightFrame.rows));
	scene_cornersRight.push_back(Point2f(rightFrame.cols, leftFrame.rows));
	scene_cornersBase.push_back(Point2f(0.0, 0.0));
	scene_cornersBase.push_back(Point2f(baseFrame.cols, 0.0));
	scene_cornersBase.push_back(Point2f(0.0, baseFrame.rows));
	scene_cornersBase.push_back(Point2f(baseFrame.cols, baseFrame.rows));
	scene_cornersFour.push_back(Point2f(0.0, 0.0));
	scene_cornersFour.push_back(Point2f(fourFrame.cols, 0.0));
	scene_cornersFour.push_back(Point2f(0.0, fourFrame.rows));
	scene_cornersFour.push_back(Point2f(fourFrame.cols, fourFrame.rows));
	scene_cornersFive.push_back(Point2f(0.0, 0.0));
	scene_cornersFive.push_back(Point2f(fiveFrame.cols, 0.0));
	scene_cornersFive.push_back(Point2f(0.0, fiveFrame.rows));
	scene_cornersFive.push_back(Point2f(fiveFrame.cols, fiveFrame.rows));
	//scene_cornersSix.push_back(Point2f(0.0, 0.0));
	//scene_cornersSix.push_back(Point2f(sixFrame.cols, 0.0));
	//scene_cornersSix.push_back(Point2f(0.0, sixFrame.rows));
	//scene_cornersSix.push_back(Point2f(sixFrame.cols, sixFrame.rows));

	perspectiveTransform(scene_cornersBase, scene_cornersBase, trans);
	perspectiveTransform(scene_cornersLeft, scene_cornersLeft, HL);
	perspectiveTransform(scene_cornersRight, scene_cornersRight, HR);
	perspectiveTransform(scene_cornersFour, scene_cornersFour, H4);
	perspectiveTransform(scene_cornersFive, scene_cornersFive, H5);
	//perspectiveTransform(scene_cornersSix, scene_cornersSix, H6);


	//Store useful information for Image limits
	int leftLimit, baseLeftLimit, baseRightLimit, rightLimit, fourLimit, fifthLimit, sixLimit;

	fifthLimit = scene_cornersFive[1].x;
	//sixLimit = scene_cornersSix[1].x;
	leftLimit = scene_cornersLeft[1].x;
	baseLeftLimit = x_offset;
	baseRightLimit = x_offset + baseFrame.cols;
	rightLimit = scene_cornersRight[0].x;
	fourLimit = scene_cornersFour[0].x;
	Mat croppedImage;
	//dfd
	//for cropping final result PLEASE REDO
	cv::Point topLeft, topRight, bottomLeft, bottomRight;
	int bottomLowerHeight, rightSmallerWidth, croppedWidth, croppedHeight;
	topLeft.y = scene_cornersFive[0].y;
	topLeft.x = scene_cornersFive[0].x;
	topRight.y = scene_cornersFour[1].y;
	topRight.x = scene_cornersFour[1].x;
	bottomLeft.y = scene_cornersFive[2].y;
	bottomLeft.x = scene_cornersFive[2].x;
	bottomRight.y = scene_cornersFour[3].y;
	bottomRight.x = scene_cornersFour[3].x;

	if (topLeft.y < 0)
		topLeft.y = 0;
	if (topLeft.x < 0)
		topLeft.x = 0;
	if (topRight.y < 0)
		topRight.y = 0;
	if (topRight.x > result.cols)
		topRight.x = result.cols;
	if (bottomRight.y > result.rows)
		bottomRight.y = result.rows;
	if (bottomRight.x > result.cols)
		bottomRight.x = result.cols;
	if (bottomLeft.y > result.rows)
		bottomLeft.y = result.rows;
	if (bottomLeft.x < 0)
		bottomLeft.x = 0;

	(bottomLeft.y < bottomRight.y) ? bottomLowerHeight = bottomLeft.y : bottomLowerHeight = bottomRight.y;
	(topRight.x < bottomRight.x) ? rightSmallerWidth = topRight.x : rightSmallerWidth = bottomRight.x;
	(topLeft.x < bottomLeft.x) ? topLeft.x = bottomLeft.x : topLeft.x = topLeft.x;
	(topLeft.y < topRight.y) ? topLeft.y = topRight.y : topLeft.y = topLeft.y;
	croppedWidth = rightSmallerWidth - topLeft.x;
	croppedHeight = bottomLowerHeight - topLeft.y;

	//Timer Info
	int _frequency = getTickFrequency();
	float _secsPerCycle = (float)1 / _frequency;
	int frameNo = 0;
	float _totalSPF = 0;

	//Get GPU ready
	cv::gpu::setDevice(0);
	cv::Mat tmp;
	capL.read(tmp);
	cv::gpu::GpuMat templ_d(tmp); // Warm up the cores

	//Initialize needed variables for GPU
	cv::gpu::GpuMat imageBSrc, imageBDst, imageRSrc, imageRDst, imageLSrc, image4Src, image4Dst, image5Src, image5Dst, imageLDst, image6Dst, image6Src;
	cv::Mat outLeftFrame, outRightFrame, outBaseFrame, outFourFrame, outFiveFrame, outSixFrame;
	outLeftFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outRightFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outBaseFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outFourFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outFiveFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	outSixFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	cv::gpu::Stream streamL, streamR, streamB, stream4, stream5, stream6;

	outputVideo.open(camResultOutput, -1, 30, cv::Size(croppedWidth, croppedHeight), true);

	//Start processing
	while (1)
	{
		frameNo++;
		int _startWhileLoop = (int)getTickCount();
		int success = capL.read(leftFrame);
		capR.read(rightFrame);
		capB.read(baseFrame);
		cap4.read(fourFrame);
		cap5.read(fiveFrame);
#ifdef ForceColorPixels	//ADDED to test RGB pixels can be removed immedietlly
		cv::circle(baseFrame, cv::Point(baseFrame.cols / 2, baseFrame.rows / 2), 5, cv::Scalar(0, 255, 0), 5);
		cv::circle(rightFrame, cv::Point(rightFrame.cols / 2, rightFrame.rows / 2), 5, cv::Scalar(0, 255, 0), 2);
		cv::circle(leftFrame, cv::Point(leftFrame.cols / 2, leftFrame.rows / 2), 5, cv::Scalar(0, 255, 0), 2);
		cv::circle(fourFrame, cv::Point(fourFrame.cols / 2, fourFrame.rows / 2), 5, cv::Scalar(0, 255, 0), 2);
		cv::circle(fiveFrame, cv::Point(fiveFrame.cols / 2, fiveFrame.rows / 2), 5, cv::Scalar(0, 255, 0), 2);
#endif



		if (!success)
			break;
		/// JH: Trasnform color space when  grayscale enabled
		if (useGrayScale) {
			cvtColor(rightFrame, rightFrame, CV_RGB2GRAY);
			cvtColor(baseFrame, baseFrame, CV_RGB2GRAY);
			cvtColor(leftFrame, leftFrame, CV_RGB2GRAY);
			cvtColor(fourFrame, fourFrame, CV_RGB2GRAY);
			cvtColor(fiveFrame, fiveFrame, CV_RGB2GRAY);
		}
		/*
		//cap6.read(sixFrame);
		resize(leftFrame, leftFrame, cv::Size(frameWidth, frameHeight));
		resize(baseFrame, baseFrame, cv::Size(frameWidth, frameHeight));
		resize(rightFrame, rightFrame, cv::Size(frameWidth, frameHeight));
		resize(fourFrame, fourFrame, cv::Size(frameWidth, frameHeight));
		resize(fiveFrame, fiveFrame, cv::Size(frameWidth, frameHeight));



		//cvtColor(sixFrame, sixFrame, CV_RGB2GRAY);
		resize(leftFrame, leftFrame, cv::Size(frameWidth, frameHeight));
		resize(baseFrame, baseFrame, cv::Size(frameWidth, frameHeight));
		resize(rightFrame, rightFrame, cv::Size(frameWidth, frameHeight));
		resize(fourFrame, fourFrame, cv::Size(frameWidth, frameHeight));
		resize(fiveFrame, fiveFrame, cv::Size(frameWidth, frameHeight));
		//resize(sixFrame, sixFrame, cv::Size(frameWidth, frameHeight));
		cv::transpose(baseFrame, baseFrame);
		cv::transpose(rightFrame, rightFrame);
		cv::transpose(leftFrame, leftFrame);
		cv::transpose(fourFrame, fourFrame);
		cv::transpose(fiveFrame, fiveFrame);
		//cv::transpose(sixFrame, sixFrame);
		cv::flip(baseFrame, baseFrame, 1);
		cv::flip(rightFrame, rightFrame, 1);
		cv::flip(leftFrame, leftFrame, 1);
		cv::flip(fourFrame, fourFrame, 1);
		cv::flip(fiveFrame, fiveFrame, 1);
		//cv::flip(sixFrame, sixFrame, 1);
		*/
		undistort(leftFrame, undistortedLeftFrame, leftIntrinsic, leftDistCoeffs);
		undistort(baseFrame, undistortedBaseFrame, baseIntrinsic, baseDistCoeffs);
		undistort(rightFrame, undistortedRightFrame, rightIntrinsic, rightDistCoeffs);
		undistort(fourFrame, undistortedFourFrame, fourIntrinsic, fourDistCoeffs);
		undistort(fiveFrame, undistortedFiveFrame, fiveIntrinsic, fiveDistCoeffs);
		//undistort(sixFrame, undistortedSixFrame, sixIntrinsic, sixDistCoeffs);

		cv::Mat rectLinearBaseFrame = rectlinearProject(undistortedBaseFrame, 0, CAM_F_MAP[BASE_CAM]);
		cv::Mat rectLinearRightFrame = rectlinearProject(undistortedRightFrame, 0, CAM_F_MAP[RIGHT_CAM]);
		cv::Mat rectLinearLeftFrame = rectlinearProject(undistortedLeftFrame, 0, CAM_F_MAP[LEFT_CAM]);
		cv::Mat rectLinearFourFrame = rectlinearProject(undistortedFourFrame, 0, CAM_F_MAP[FOUR_CAM]);
		cv::Mat rectLinearFiveFrame = rectlinearProject(undistortedFiveFrame, 0, CAM_F_MAP[FIFTH_CAM]);
		//cv::Mat rectLinearSixFrame = rectlinearProject(undistortedSixFrame, 0, CAM_F_MAP[BACK_CAM]);

		baseFrame = rectLinearBaseFrame;
		leftFrame = rectLinearLeftFrame;
		rightFrame = rectLinearRightFrame;
		fourFrame = rectLinearFourFrame;
		fiveFrame = rectLinearFiveFrame;
		//sixFrame = rectLinearSixFrame;

		//Upload back to GPU
		streamL.enqueueUpload(leftFrame, imageLSrc);
		streamR.enqueueUpload(rightFrame, imageRSrc);
		streamB.enqueueUpload(baseFrame, imageBSrc);
		stream4.enqueueUpload(fourFrame, image4Src);
		stream5.enqueueUpload(fiveFrame, image5Src);
		//stream6.enqueueUpload(sixFrame, image6Src);

		//Warp Perspective
		gpu::warpPerspective(imageBSrc, resultB, trans, cv::Size(resultHeight + 600, resultWidth), cv::INTER_NEAREST | CV_WARP_FILL_OUTLIERS);
		gpu::warpPerspective(imageRSrc, resultR, HR, cv::Size(resultHeight + 600, resultWidth), cv::INTER_NEAREST | CV_WARP_FILL_OUTLIERS);
		gpu::warpPerspective(imageLSrc, resultL, HL, cv::Size(resultHeight + 600, resultWidth), cv::INTER_NEAREST | CV_WARP_FILL_OUTLIERS);
		gpu::warpPerspective(image4Src, result4, H4, cv::Size(resultHeight + 600, resultWidth), cv::INTER_NEAREST | CV_WARP_FILL_OUTLIERS);
		gpu::warpPerspective(image5Src, result5, H5, cv::Size(resultHeight + 600, resultWidth), cv::INTER_NEAREST | CV_WARP_FILL_OUTLIERS);
		//gpu::warpPerspective(image6Src, result6, H6, cv::Size(resultHeight + 600, resultWidth), cv::INTER_NEAREST | CV_WARP_FILL_OUTLIERS);

		streamL.enqueueDownload(resultL, outLeftFrame);
		streamR.enqueueDownload(resultR, outRightFrame);
		streamB.enqueueDownload(resultB, outBaseFrame);
		stream4.enqueueDownload(result4, outFourFrame);
		stream5.enqueueDownload(result5, outFiveFrame);
		//stream6.enqueueDownload(result6, outSixFrame);

		//streamL.waitForCompletion();
		if (!useGrayScale) {
			/// JH: Support of RGB images when useGrayScale is disabled
			std::cout << "RGB image used" << std::endl;
			std::cout << outFiveFrame.type() << " " << CV_8UC3 << std::endl;
			for (int j = 0; j < result.rows; ++j)
				for (int i = 0; i < result.cols; ++i)
				{
					//cout << "blending" << endl;
					/**
					cv::Vec3b cL(0, 0, 0);
					cv::Vec3b cB(0, 0, 0);
					cv::Vec3b cR(0, 0, 0);
					cv::Vec3b cLB(0, 0, 0);
					cv::Vec3b cBR(0, 0, 0);
					cv::Vec3b color(0, 0, 0);
					*/
					float blendA = 0.8;
					cv::Vec3b cL;
					cv::Vec3b cB;
					cv::Vec3b cR;
					cv::Vec3b c4;
					cv::Vec3b c5;
					cv::Vec3b c6;
					cv::Vec3b cLB;
					cv::Vec3b cBR;
					cv::Vec3b cR4;
					cv::Vec3b c5L;
					cv::Vec3b c65;
					cv::Vec3b color;

					float coeff = 0.4;
					int blendValue;
					bool cL_0 = false;
					bool cB_0 = false;
					bool cR_0 = false;
					bool c4_0 = false;
					bool c5_0 = false;
					bool c6_0 = false;


					//color = resultB.at<uchar>(j, i) + resultL.at<uchar>(j, i) + resultR.at<uchar>(j, i) + result4.at<uchar>(j, i) + result5.at<uchar>(j, i);


					// Assign flags
					/*
					if (j < result.rows && i < sixLimit){
					c6_0 = true;
					c6 = outSixFrame.at<uchar>(j, i);
					}
					*/
					if (j < result.rows && i < fifthLimit){
						c5_0 = true;
						c5 = outFiveFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i < leftLimit && i > fifthLimit){
						cL_0 = true;
						cL = outLeftFrame.at<cv::Vec3b>(j, i);
					}
					if (j < baseFrame.rows && i>baseLeftLimit && i < baseRightLimit) {
						//cout << "cB is true" << endl;
						cB_0 = true;
						cB = outBaseFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i>rightLimit && i < fourLimit) {
						cR_0 = true;
						cR = outRightFrame.at<cv::Vec3b>(j, i);
					}
					if (j < result.rows && i> fourLimit) {
						c4_0 = true;
						c4 = outFourFrame.at<cv::Vec3b>(j, i);
					}


					// Activate color based on flags
					if (c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// Use combination of five + left
						color = ((1 - blendA)*cL + blendA*c5);
					}
					else if (!c5_0 && cL_0 && cB_0 && !cR_0)
					{
						// Use combination of base + left
						color = ((1 - blendA)*cL + blendA*cB);
					}
					else if (!c5_0 && !cL_0 && cB_0 && cR_0 && !c4_0)
					{
						// Use combination of base + right
						color = ((1 - blendA)*cB + blendA*cR);
					}
					else if (!c5_0 && !cL_0 && !cB_0 && cR_0 && c4_0)
					{
						// Use combination of four + right
						color = ((1 - blendA)*cR + blendA*c4);
					}
					/*
					else if (c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// Use combination of six + five
					color = ((1 - blendA)*c5 + blendA*c6);
					}
					*/
					else if (!c6_0 && !c5_0 && !cL_0 && cB_0 && !cR_0 && !c4_0)
					{
						// In base's frame
						color = cB;
					}
					else if (!c6_0 && !c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// In left's frame
						color = cL;
					}

					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && cR_0 && !c4_0)
					{
						// In right frame
						color = cR;
					}
					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && c4_0)
					{
						// In fourth frame
						color = c4;
					}
					else if (!c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
						// In fifth frame
						color = c5;
					}
					/*
					else if (c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// In sixth frame
					color = c6;
					}
					*/
					//result.at<cv::Vec3b>(j, i) = color;
					result.at<cv::Vec3b>(j, i) = color;

				}

		}
		else {
			for (int j = 0; j < result.rows; ++j)
				for (int i = 0; i < result.cols; ++i)
				{
					//cout << "blending" << endl;
					/**
					cv::Vec3b cL(0, 0, 0);
					cv::Vec3b cB(0, 0, 0);
					cv::Vec3b cR(0, 0, 0);
					cv::Vec3b cLB(0, 0, 0);
					cv::Vec3b cBR(0, 0, 0);
					cv::Vec3b color(0, 0, 0);
					*/
					float blendA = 0.8;
					uchar cL;
					uchar cB;
					uchar cR;
					uchar c4;
					uchar c5;
					uchar c6;
					uchar cLB;
					uchar cBR;
					uchar cR4;
					uchar c5L;
					uchar c65;
					uchar color;

					float coeff = 0.4;
					int blendValue;
					bool cL_0 = false;
					bool cB_0 = false;
					bool cR_0 = false;
					bool c4_0 = false;
					bool c5_0 = false;
					bool c6_0 = false;


					//color = resultB.at<uchar>(j, i) + resultL.at<uchar>(j, i) + resultR.at<uchar>(j, i) + result4.at<uchar>(j, i) + result5.at<uchar>(j, i);


					// Assign flags
					/*
					if (j < result.rows && i < sixLimit){
					c6_0 = true;
					c6 = outSixFrame.at<uchar>(j, i);
					}
					*/
					if (j < result.rows && i < fifthLimit){
						c5_0 = true;
						c5 = outFiveFrame.at<uchar>(j, i);
					}
					if (j < result.rows && i < leftLimit && i > fifthLimit){
						cL_0 = true;
						cL = outLeftFrame.at<uchar>(j, i);
					}
					if (j < baseFrame.rows && i>baseLeftLimit && i < baseRightLimit) {
						//cout << "cB is true" << endl;
						cB_0 = true;
						cB = outBaseFrame.at<uchar>(j, i);
					}
					if (j < result.rows && i>rightLimit && i < fourLimit) {
						cR_0 = true;
						cR = outRightFrame.at<uchar>(j, i);
					}
					if (j < result.rows && i> fourLimit) {
						c4_0 = true;
						c4 = outFourFrame.at<uchar>(j, i);
					}


					// Activate color based on flags
					if (c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// Use combination of five + left
						color = ((1 - blendA)*cL + blendA*c5);
					}
					else if (!c5_0 && cL_0 && cB_0 && !cR_0)
					{
						// Use combination of base + left
						color = ((1 - blendA)*cL + blendA*cB);
					}
					else if (!c5_0 && !cL_0 && cB_0 && cR_0 && !c4_0)
					{
						// Use combination of base + right
						color = ((1 - blendA)*cB + blendA*cR);
					}
					else if (!c5_0 && !cL_0 && !cB_0 && cR_0 && c4_0)
					{
						// Use combination of four + right
						color = ((1 - blendA)*cR + blendA*c4);
					}
					/*
					else if (c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// Use combination of six + five
					color = ((1 - blendA)*c5 + blendA*c6);
					}
					*/
					else if (!c6_0 && !c5_0 && !cL_0 && cB_0 && !cR_0 && !c4_0)
					{
						// In base's frame
						color = cB;
					}
					else if (!c6_0 && !c5_0 && cL_0 && !cB_0 && !cR_0)
					{
						// In left's frame
						color = cL;
					}

					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && cR_0 && !c4_0)
					{
						// In right frame
						color = cR;
					}
					else if (!c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && c4_0)
					{
						// In fourth frame
						color = c4;
					}
					else if (!c6_0 && c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
						// In fifth frame
						color = c5;
					}
					/*
					else if (c6_0 && !c5_0 && !cL_0 && !cB_0 && !cR_0 && !c4_0)
					{
					// In sixth frame
					color = c6;
					}
					*/
					//result.at<cv::Vec3b>(j, i) = color;
					result.at<uchar>(j, i) = color;

				}
		}
		croppedImage = result(Rect(topLeft.x, topLeft.y, croppedWidth, croppedHeight));

		cv::imshow("left", outLeftFrame);
		cv::imshow("right", outRightFrame);
		cv::imshow("base", outBaseFrame);
		cv::imshow("four", outFourFrame);
		cv::imshow("five", outFiveFrame);
		//cv::imshow("Result", result);
		//imshow("cropped Result", croppedImage);

		outputVideo << croppedImage;

		//Latency Calculations
		int _endWhileLoop = (int)getTickCount();
		int _WhileLoopDiff = _endWhileLoop - _startWhileLoop;
		float _secsForWhileLoop = (float)(_secsPerCycle * _WhileLoopDiff);
		cout << "secs for Frame " << frameNo << " is " << _secsForWhileLoop << endl;
		_totalSPF = _totalSPF + _secsForWhileLoop;

		if ((frameNo % 30) == 0)
		{
			float _aveSPF = (float)_totalSPF / 30.0;
			cout << "Average Seconds-Per-Frame for past 30 frames is: " << _aveSPF << endl;
			_totalSPF = 0;
		}

		if (waitKey(30) == 27)
			break;
	}



	return 0;
}


int calibrateCamerasInternal(int cam)
{
	int numBoards = 20;
	int numCornersHor = 8;
	int numCornersVer = 5;
	int cameraID = cam;
	int cameraResolutionWidth;
	int cameraResolutionHeight;

	int numSquares = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);

	VideoCapture capture = VideoCapture(cameraID);

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	cameraResolutionWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH)*0.25;
	cameraResolutionHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT)*0.25;
	//capture.set(CV_CAP_PROP_FRAME_WIDTH, cameraResolutionWidth);
	//capture.set(CV_CAP_PROP_FRAME_HEIGHT, cameraResolutionHeight);
	cout << "camera resolution [width height] is " << capture.get(CV_CAP_PROP_FRAME_WIDTH) << " and " << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;

	vector<vector<Point3f>> object_points;
	vector<vector<Point2f>> image_points;
	vector<Point2f> corners;
	int successes = 0;
	Mat image;
	Mat gray_image;

	capture >> image;
	resize(image, image, cv::Size(cameraResolutionWidth, cameraResolutionHeight));
	transpose(image, image);
	flip(image, image, 1);
	vector<Point3f> obj;
	for (int j = 0; j<numSquares; j++)
		obj.push_back(Point3f(j / numCornersHor, j%numCornersHor, 0.0f));

	while (successes<numBoards)
	{

		cvtColor(image, gray_image, CV_BGR2GRAY);
		bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found)
		{
			cornerSubPix(gray_image, corners, Size(9, 9), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray_image, board_sz, corners, found);
		}

		//cout << image.size() << endl;
		imshow("win1", image);
		imshow("win2", gray_image);

		capture >> image;
		resize(image, image, cv::Size(cameraResolutionWidth, cameraResolutionHeight));
		transpose(image, image);
		flip(image, image, 1);

		int key = waitKey(1);

		if (key == 27)

			return 0;

		if (key == ' ' && found != 0)
		{
			image_points.push_back(corners);
			object_points.push_back(obj);

			printf("Snap stored!");

			successes++;

			if (successes >= numBoards)
				break;
		}

	}

	Mat intrinsic = Mat(3, 3, CV_32FC1);
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	intrinsic.ptr<float>(0)[0] = 1;
	intrinsic.ptr<float>(1)[1] = 1;

	calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);

	if (cameraID == BASE_CAM)
	{
		cout << "saving to base" << endl;
		cv::FileStorage file;
		file.open("intrinsic-base.txt", cv::FileStorage::WRITE);
		file << "Intrinsic Matrix" << intrinsic;
		file.open("distortion-base.txt", cv::FileStorage::WRITE);
		file << "Distortion Matrix" << distCoeffs;
	}
	else if (cameraID == RIGHT_CAM)
	{
		cout << "saving to right" << endl;
		cv::FileStorage file;
		file.open("intrinsic-right.txt", cv::FileStorage::WRITE);
		file << "Intrinsic Matrix" << intrinsic;
		file.open("distortion-right.txt", cv::FileStorage::WRITE);
		file << "Distortion Matrix" << distCoeffs;
	}
	else if (cameraID == LEFT_CAM)
	{
		cout << "saving to left" << endl;
		cv::FileStorage file;
		file.open("intrinsic-left.txt", cv::FileStorage::WRITE);
		file << "Intrinsic Matrix" << intrinsic;
		file.open("distortion-left.txt", cv::FileStorage::WRITE);
		file << "Distortion Matrix" << distCoeffs;
	}
	else if (cameraID == FOUR_CAM)
	{
		cout << "saving to fourth" << endl;
		cv::FileStorage file;
		file.open("intrinsic-four.txt", cv::FileStorage::WRITE);
		file << "Intrinsic Matrix" << intrinsic;
		file.open("distortion-four.txt", cv::FileStorage::WRITE);
		file << "Distortion Matrix" << distCoeffs;
	}
	else if (cameraID == FIFTH_CAM)
	{
		cout << "saving to fifth" << endl;
		cv::FileStorage file;
		file.open("intrinsic-five.txt", cv::FileStorage::WRITE);
		file << "Intrinsic Matrix" << intrinsic;
		file.open("distortion-five.txt", cv::FileStorage::WRITE);
		file << "Distortion Matrix" << distCoeffs;
	}
	else if (cameraID == BACK_CAM)
	{
		cout << "saving to sixth" << endl;
		cv::FileStorage file;
		file.open("intrinsic-six.txt", cv::FileStorage::WRITE);
		file << "Intrinsic Matrix" << intrinsic;
		file.open("distortion-six.txt", cv::FileStorage::WRITE);
		file << "Distortion Matrix" << distCoeffs;
	}


	cout << "intrinsic matrix is " << intrinsic << endl;
	cout << "dist matrix is " << distCoeffs << endl;

	Mat imageUndistorted;
	while (1)
	{
		capture >> image;
		resize(image, image, cv::Size(cameraResolutionWidth, cameraResolutionHeight));
		transpose(image, image);
		flip(image, image, 1);
		undistort(image, imageUndistorted, intrinsic, distCoeffs);

		imshow("win1", image);
		imshow("win2", imageUndistorted);
		waitKey(1);
	}

	capture.release();
	return 0;
}


int calibrateCamerasExternal(int baseCam, int sideCam)
{
	// Load the images
	cv::VideoCapture capL(sideCam);
	cv::VideoCapture capB(baseCam);
	capL.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	capL.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	capB.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	capB.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	int frameWidth = capL.get(CV_CAP_PROP_FRAME_WIDTH)*0.25;
	int frameHeight = capL.get(CV_CAP_PROP_FRAME_HEIGHT)*0.25;
	/*
	cv::Mat baseIntrinsic = Mat(3, 3, CV_32FC1);
	cv::Mat rightIntrinsic = Mat(3, 3, CV_32FC1);
	cv::Mat leftIntrinsic = Mat(3, 3, CV_32FC1);
	cv::Mat fourIntrinsic = Mat(3, 3, CV_32FC1);
	cv::Mat fiveIntrinsic = Mat(3, 3, CV_32FC1);
	*/
	cv::Mat undistortedBaseFrame, undistortedRightFrame, undistortedLeftFrame, undistortedFourFrame, undistortedFiveFrame;
	FileStorage file;

	cout << "Press 'n' to take picture and calibrate" << endl;
	while (1)
	{
		capL.read(rightFrame);
		capB.read(baseFrame);


		cvtColor(rightFrame, rightFrame, CV_RGB2GRAY);
		cvtColor(baseFrame, baseFrame, CV_RGB2GRAY);
		resize(rightFrame, rightFrame, cv::Size(frameWidth, frameHeight));
		resize(baseFrame, baseFrame, cv::Size(frameWidth, frameHeight));
	cv:transpose(baseFrame, baseFrame);
		cv::transpose(rightFrame, rightFrame);
		cv::flip(baseFrame, baseFrame, 1);
		cv::flip(rightFrame, rightFrame, 1);

		if (baseCam == BASE_CAM)
		{
			cout << "base is base_cam" << endl;
			undistort(baseFrame, undistortedBaseFrame, baseIntrinsic, baseDistCoeffs);
		}
		else if (baseCam == LEFT_CAM)
		{
			cout << "base is left cam" << endl;
			undistort(baseFrame, undistortedBaseFrame, leftIntrinsic, leftDistCoeffs);
		}
		else if (baseCam == RIGHT_CAM)
		{
			cout << "base is right cam" << endl;
			undistort(baseFrame, undistortedBaseFrame, rightIntrinsic, rightDistCoeffs);
		}
		else if (baseCam == FIFTH_CAM)
		{
			cout << "base is fifth cam" << endl;
			undistort(baseFrame, undistortedBaseFrame, fiveIntrinsic, fiveDistCoeffs);
		}



		if (sideCam == RIGHT_CAM)
		{
			cout << "undistorting right cam" << endl;
			undistort(rightFrame, undistortedRightFrame, rightIntrinsic, rightDistCoeffs);
		}
		else if (sideCam == LEFT_CAM)
		{
			cout << "undistorting left cam" << endl;
			undistort(rightFrame, undistortedRightFrame, leftIntrinsic, leftDistCoeffs);
		}
		else if (sideCam == FOUR_CAM)
		{
			cout << "undistorting fourth cam" << endl;
			undistort(rightFrame, undistortedRightFrame, fourIntrinsic, fourDistCoeffs);
		}
		else if (sideCam == FIFTH_CAM)
		{
			cout << "undistorting fifth cam" << endl;
			undistort(rightFrame, undistortedRightFrame, fiveIntrinsic, fiveDistCoeffs);
		}
		else if (sideCam == BACK_CAM)
		{
			cout << "undistorting back cam" << endl;
			undistort(rightFrame, undistortedRightFrame, sixIntrinsic, sixDistCoeffs);
		}






		cout << "baseCam F length is: " << CAM_F_MAP[baseCam] << endl;
		cout << "sideCam F length is: " << CAM_F_MAP[sideCam] << endl;

		cv::Mat rectLinearBaseFrame = rectlinearProject(undistortedBaseFrame, 0, CAM_F_MAP[baseCam]);
		cv::Mat rectLinearRightFrame = rectlinearProject(undistortedRightFrame, 0, CAM_F_MAP[sideCam]);


		baseFrame = rectLinearBaseFrame;
		rightFrame = rectLinearRightFrame;

		/*
		cvtColor(undistortedBaseFrame, undistortedBaseFrame, CV_BGR2GRAY);
		bool foundBase = findChessboardCorners(undistortedBaseFrame, board_sz, baseImage, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		cvtColor(undistortedRightFrame, undistortedRightFrame, CV_BGR2GRAY);
		bool foundRight = findChessboardCorners(undistortedRightFrame, board_sz, rightImage, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		cout << "FoundBase and FoundRight is: " << foundBase << " "<< foundRight << endl;
		if(foundBase && foundRight)
		{
		cornerSubPix(undistortedBaseFrame, baseImage, Size(9, 9), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
		cornerSubPix(undistortedRightFrame, rightImage, Size(9, 9), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));

		drawChessboardCorners(undistortedBaseFrame, board_sz, baseImage, foundBase);
		drawChessboardCorners(undistortedRightFrame, board_sz, rightImage, foundBase);

		}

		imshow("Undistorted Right Image", undistortedRightFrame);
		imshow("Undistorted Base Image", undistortedBaseFrame);

		if ((waitKey(30) == 110) && (foundBase == 1) && (foundRight == 1))
		{
		baseCorners = baseImage;
		rightCorners = rightImage;
		successes++;
		printf("Snap stored!");
		if(successes>=numBoards)
		break;
		}
		if(waitKey(30) == 27)
		break;
		*/

		imshow("Side Image", rightFrame);
		imshow("Base Image", baseFrame);
		//makePanorama(frame1, frame2, 1);
		if (waitKey(100) == 110)
			break;
	}

	Mat right = rightFrame;
	Mat base = baseFrame;
	Mat gray_image1;
	Mat gray_image2;

	imshow("Side Image", rightFrame);
	imshow("Base Image", baseFrame);

	cout << "Checking images..." << endl;
	cout << "Press any Key to move forward..." << endl;
	waitKey(0);

	cv::Mat H;
	//set the callback function for any mouse event
	baseImage.clear();
	rightImage.clear();

	cout << "Input matching points..." << endl;
	cout << "Once done, press any Key to move forward..." << endl;
	setMouseCallback("Base Image", BaseCallBackFunc, NULL);
	setMouseCallback("Side Image", RightCallBackFunc2, NULL);

	// Wait until user press some key
	waitKey(0);

	std::cout << "Base Image Array: " << baseImage << std::endl;
	std::cout << "Right Image Array: " << rightImage << std::endl;

	// Find the Homography Matrix
	H = findHomography(rightImage, baseImage, CV_RANSAC);

	//H = cv::getPerspectiveTransform(rightImage, baseImage);
	//H = estimateRigidTransform(rightImage, baseImage, CV_RANSAC);

	//cv::FileStorage file;
	if (sideCam == RIGHT_CAM)
	{
		cout << "Saving to right..." << endl;
		file.open(externalCalibrationPath + "/H-right.txt", cv::FileStorage::WRITE);
	}
	else if (sideCam == LEFT_CAM)
	{
		cout << "Saving to left..." << endl;
		file.open(externalCalibrationPath + "/H-left.txt", cv::FileStorage::WRITE);
	}
	else if (sideCam == FOUR_CAM)
	{
		cout << "Saving to fourth..." << endl;
		file.open(externalCalibrationPath + "/H-four.txt", cv::FileStorage::WRITE);
	}
	else if (sideCam == FIFTH_CAM)
	{
		cout << "Saving to fifth..." << endl;
		file.open(externalCalibrationPath + "/H-five.txt", cv::FileStorage::WRITE);
	}

	file << "H Matrix" << H;
	std::cout << "Homography matrix saved" << std::endl;
	waitKey(0);

	return 1;
}