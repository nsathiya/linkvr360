#include "Test.h"

using namespace std;
using namespace cv;

Test::Test(bool showPic)
{
	testPic = showPic;
}


Test::~Test()
{
}

int Test::getWorld(){

	if(testPic){
		// std::cout << "Test1" << std::endl;
		// cv::Mat ten = cv::Mat(500, 500, CV_8SC3);
		// std::cout << "Test2" << std::endl;
		// ten.setTo(cv::Scalar(200, 100, 5));
		// std::cout << "Test3" << std::endl;
		// std::cout << ten << std::endl;
		// std::cout << "Test4" << std::endl;
		// cv::imwrite("ten.jpg", ten);

		std::cout << cv::getBuildInformation() << std::endl;

		char cCurrentPath[FILENAME_MAX];

		if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath)))
		{
			return errno;
		}

		cCurrentPath[sizeof(cCurrentPath) - 1] = '\0'; /* not really required */

		printf("The current working directory is %s", cCurrentPath);




		/*std::ifstream myReadFile;
		myReadFile.open("./test.txt");
		char output[100];
		std::cout << "READING" << std::endl;
		if (myReadFile.is_open()) {
			std::cout << "OPEN" << std::endl;
			while (!myReadFile.eof()) {
				myReadFile >> output;
				std::cout<<output;
			}
		}
		myReadFile.close();

		cv::Mat ten = cv::imread("ten.png", CV_8SC3);

		cv::Mat matToAdd = cv::Mat(ten.rows, ten.cols, ten.type());
		matToAdd.setTo(cv::Scalar(50, 25, 100));
		std::cout << ten.type() << std::endl;
		std::cout << matToAdd.type() << std::endl;

		cv::imshow("ten before add", ten);
		cv::waitKey(0);

		cv::add(ten, matToAdd, ten);
		cv::imshow("ten after add", ten);
		cv::waitKey(0);
		cv::imwrite("ten.png", ten);*/

		setup();
		testingFunction(true, true, false);
	}
	
	return 0;
}

const int BASE_CAM = 1;  //0;
const int LEFT_CAM = 0;  //2;  //1; // 3;
const int RIGHT_CAM = 2; // 1; // 2; // 4;
const int FOUR_CAM = 1;
const int FIFTH_CAM = 2;
const int BACK_CAM = 5;
const int NO_OF_CAMS = 3;

std::string videoPath = "samples";
std::string preprocess = videoPath + "/preprocess";
std::string calibrationPath = "calibration";
std::string internalCalibrationPath = calibrationPath + "/internal";
std::string externalCalibrationPath = calibrationPath + "/external";
std::string camLOutput = preprocess + "/Cam_L_Stream.avi";
std::string camROutput = preprocess + "/Cam_R_Stream.avi";
std::string camBOutput = preprocess + "/Cam_B_Stream.avi";
std::string cam4Output = preprocess + "/Cam_4_Stream.avi";
std::string cam5Output = preprocess + "/Cam_5_Stream.avi";
std::string camResultOutput = videoPath + "/results/Cam_Result_Stream.avi";
std::string camPicPrefix = preprocess + "/CamPic";

// Distortion
std::string H_File, IntrinsicBase_File, DistRight_File, IntrinsicRight_File, DistBase_File, IntrinsicLeft_File, DistLeft_File, IntrinsicFour_File, DistFour_File, IntrinsicFive_File, DistFive_File, IntrinsicSix_File, DistSix_File;
cv::Mat baseIntrinsic = cv::Mat(3, 3, CV_32FC1);
cv::Mat rightIntrinsic = cv::Mat(3, 3, CV_32FC1);
cv::Mat leftIntrinsic = cv::Mat(3, 3, CV_32FC1);
cv::Mat fourIntrinsic = cv::Mat(3, 3, CV_32FC1);
cv::Mat fiveIntrinsic = cv::Mat(3, 3, CV_32FC1);
cv::Mat sixIntrinsic = cv::Mat(3, 3, CV_32FC1);
cv::Mat baseDistCoeffs, rightDistCoeffs, leftDistCoeffs, fourDistCoeffs, fiveDistCoeffs, sixDistCoeffs;
cv::Mat HR = cv::Mat(3, 3, CV_32FC1);
cv::Mat HL = cv::Mat(3, 3, CV_32FC1);
cv::Mat H4 = cv::Mat(3, 3, CV_32FC1);
cv::Mat H5 = cv::Mat(3, 3, CV_32FC1);
cv::Mat H6 = cv::Mat(3, 3, CV_32FC1);
std::string H_R, H_L, H_4, H_5, H_6;
std::map<int, float> CAM_F_MAP;
std::vector<cv::Point2f> baseImage;
std::vector<cv::Point2f> rightImage;
cv::Mat rightFrame, baseFrame;

/// Control grayScale option
bool useGrayScale = true;
cv::Mat border(cv::Mat mask);
std::vector<cv::Mat> FRAMES(NO_OF_CAMS);
std::vector<cv::Mat> INTRINSICCOEFFS(NO_OF_CAMS);
std::vector<cv::Mat> EXTRINSICCOEFFS(NO_OF_CAMS);
std::vector<cv::Mat> DISTORTIONCOEFFS(NO_OF_CAMS);
std::vector<cv::Mat> RESULTS(NO_OF_CAMS);
std::vector<std::string> PREPROCESS_FRAMES_PATH(NO_OF_CAMS);
std::vector<std::string> RESULTS_FRAMES_PATH(1);
std::vector<float> FOCAL(NO_OF_CAMS);

void Test::setup()
{

	FOCAL[0] = CAM_F_MAP[BASE_CAM] = 395.164;
	FOCAL[1] = CAM_F_MAP[LEFT_CAM] = 422.400;
	FOCAL[2] = CAM_F_MAP[RIGHT_CAM] = 326.38;
	//FOCAL[3] = CAM_F_MAP[FOUR_CAM] = 176.9511;
	//FOCAL[4] = CAM_F_MAP[FIFTH_CAM] = 175.2695;
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
	//INTRINSICCOEFFS[3] = fourIntrinsic;
	//INTRINSICCOEFFS[4] = fiveIntrinsic;

	DISTORTIONCOEFFS[0] = baseDistCoeffs;
	DISTORTIONCOEFFS[1] = leftDistCoeffs;
	DISTORTIONCOEFFS[2] = rightDistCoeffs;
	//DISTORTIONCOEFFS[3] = fourDistCoeffs;
	//DISTORTIONCOEFFS[4] = fiveDistCoeffs;

	EXTRINSICCOEFFS[0] = NULL;
	EXTRINSICCOEFFS[1] = HL;
	EXTRINSICCOEFFS[2] = HR;
	//EXTRINSICCOEFFS[3] = H4;
	//EXTRINSICCOEFFS[4] = H5;

	PREPROCESS_FRAMES_PATH[0] = camBOutput;
	PREPROCESS_FRAMES_PATH[1] = camLOutput;
	PREPROCESS_FRAMES_PATH[2] = camROutput;
	//PREPROCESS_FRAMES_PATH[0] = cam4Output;
	//PREPROCESS_FRAMES_PATH[0] = cam5Output;

	RESULTS_FRAMES_PATH[0] = camResultOutput;

}

int Test::testingFunction(bool GPU, bool stitchFromMemory, bool stitchVideo) {

	std::string method = "w/o GPU";
	if (GPU)
		method = "w GPU";

	std::cout << "Stitching " + method << std::endl;
	std::cout << "Stitching from memory: " << stitchFromMemory << std::endl;
	if (stitchFromMemory)
		std::cout << "If stitch from memory, stitch video: " << stitchVideo << std::endl;

	std::vector<int> cameraPorts(NO_OF_CAMS);
	cameraPorts[0] = BASE_CAM;
	cameraPorts[1] = LEFT_CAM;
	cameraPorts[2] = RIGHT_CAM;
	//cameraPorts[3] = FOUR_CAM;
	//cameraPorts[4] = FIFTH_CAM;
	CameraOps *CO = new CameraOps(cameraPorts, PREPROCESS_FRAMES_PATH, stitchVideo);
	ImageOps *IO = new ImageOps();
	GPUOps *GO = GPU ? new GPUOps(NO_OF_CAMS) : NULL;
	BlenderOps *BO = new BlenderOps();
	MemoryManager *MM;

	cv::VideoWriter outputVideo;

	std::vector<cv::Point2f> scene_cornersLeft, scene_cornersRight, scene_cornersBase, scene_cornersFour, scene_cornersFive, scene_cornersSix, scene_corners;

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
	
	MM->readFrames(FRAMES, camPicPrefix);
	/*imshow("frame1", FRAMES[0]);
	imshow("frame2", FRAMES[1]);
	imshow("frame3", FRAMES[2]);
	cv::waitKey(0);*/

	int frameWidth = FRAMES[0].cols; //CO->CO_getProp(CV_CAP_PROP_FRAME_WIDTH, 0); // *0.25;
	int frameHeight = FRAMES[0].rows; //CO->CO_getProp(CV_CAP_PROP_FRAME_HEIGHT, 0); // *0.25;
	int resultWidth = frameWidth * 2;
	int resultHeight = frameHeight + 100;
	bool record = false;
	cout << frameWidth << " " << frameHeight << endl;
	cv::Mat result = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	if (stitchFromMemory && !stitchVideo)
		MM = new MemoryManager(NO_OF_CAMS, PREPROCESS_FRAMES_PATH, frameHeight, frameWidth, stitchVideo);
	if (stitchFromMemory && stitchVideo)
		MM = new MemoryManager(1, RESULTS_FRAMES_PATH, resultHeight + 1000, resultWidth / 2, stitchVideo);

	// Move Scene to the right by 100
	int x_offset = 400.0;
	float y_offset = 100.0;
	float z_offset = 100.0;
	float transdata[] = { 1.0, 0.0, x_offset, 0.0, 1.0, y_offset, 0.0, 0.0, 1.0 };
	cv::Mat trans(3, 3, CV_32FC1, transdata);
	cout << "HR: " << HR << endl;
	EXTRINSICCOEFFS[0] = trans;

	cv::Mat HR_m = HR.clone();
	cv::Mat HL_m = HL.clone();
	HR = trans * HR;
	HL = trans * HL;
	H4 = trans * HR_m * H4;
	H5 = trans * HL_m * H5;
	//H6 = trans * HL_m * H5 * H6;

	cout << "finished getting cv::Matrix" << endl;

	//CO->CO_captureFrames(FRAMES);
	
	if (FRAMES[0].cols == 0) {
		cout << "Error reading file " << endl;
		return -1;
	}

	//IO->IO_resize(FRAMES, cv::Size(frameWidth, frameHeight));
	//IO->IO_transpose(FRAMES);
	//IO->IO_flip(FRAMES, 1);
	IO->IO_undistort(FRAMES, INTRINSICCOEFFS, DISTORTIONCOEFFS);

	// Use the Homography cv::Matrix to warp the images
	scene_corners.clear();
	scene_cornersBase.push_back(Point2f(0.0, 0.0));
	scene_cornersBase.push_back(Point2f(FRAMES[0].cols, 0.0));
	scene_cornersBase.push_back(Point2f(0.0, FRAMES[0].rows));
	scene_cornersBase.push_back(Point2f(FRAMES[0].cols, FRAMES[0].rows));
	scene_cornersLeft.push_back(Point2f(0.0, 0.0));
	scene_cornersLeft.push_back(Point2f(FRAMES[1].cols, 0.0));
	scene_cornersLeft.push_back(Point2f(0.0, FRAMES[1].rows));
	scene_cornersLeft.push_back(Point2f(FRAMES[1].cols, FRAMES[1].rows));
	/*scene_cornersRight.push_back(Point2f(0.0, 0.0));
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
	scene_cornersFive.push_back(Point2f(FRAMES[4].cols, FRAMES[4].rows));*/
	//scene_cornersSix.push_back(Point2f(0.0, 0.0));
	//scene_cornersSix.push_back(Point2f(sixFrame.cols, 0.0));
	//scene_cornersSix.push_back(Point2f(0.0, sixFrame.rows));
	//scene_cornersSix.push_back(Point2f(sixFrame.cols, sixFrame.rows));

	perspectiveTransform(scene_cornersBase, scene_cornersBase, trans);
	perspectiveTransform(scene_cornersLeft, scene_cornersLeft, HL);/*
																   perspectiveTransform(scene_cornersRight, scene_cornersRight, HR);
																   perspectiveTransform(scene_cornersFour, scene_cornersFour, H4);
																   perspectiveTransform(scene_cornersFive, scene_cornersFive, H5);*/
	//perspectiveTransform(scene_cornersSix, scene_cornersSix, H6);


	//Store useful inforcv::Mation for Image limits
	int leftLimit, baseLeftLimit, baseRightLimit, rightLimit, fourLimit, fifthLimit, sixLimit;

	//fifthLimit = scene_cornersFive[1].x;
	////sixLimit = scene_cornersSix[1].x;
	leftLimit = scene_cornersLeft[1].x;
	baseLeftLimit = x_offset;
	//baseRightLimit = x_offset + FRAMES[0].cols;
	//rightLimit = scene_cornersRight[0].x;
	//fourLimit = scene_cornersFour[0].x;
	cv::Mat croppedImage;
	BO->limitPt.leftXLimit = leftLimit;
	BO->limitPt.rightXLimit = baseLeftLimit;

	//for cropping final result PLEASE REDO
	cv::Point topLeft, topRight, bottomLeft, bottomRight;
	int bottomLowerHeight, rightSmallerWidth, croppedWidth, croppedHeight;
	topLeft.y = scene_cornersLeft[0].y;
	topLeft.x = scene_cornersLeft[0].x;
	topRight.y = scene_cornersBase[1].y;
	topRight.x = scene_cornersBase[1].x;
	bottomLeft.y = scene_cornersLeft[2].y;
	bottomLeft.x = scene_cornersLeft[2].x;
	bottomRight.y = scene_cornersBase[3].y;
	bottomRight.x = scene_cornersBase[3].x;

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

	//Initialize needed variables for GPU
	RESULTS[0] = cv::Mat(resultWidth, resultHeight + 1000, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[1] = cv::Mat(resultWidth, resultHeight + 1000, useGrayScale ? CV_8UC1 : CV_8UC3);
	RESULTS[2] = cv::Mat(resultWidth, resultHeight + 1000, useGrayScale ? CV_8UC1 : CV_8UC3);
	//RESULTS[3] = cv::cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	//RESULTS[4] = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	//outSixFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	//Start processing
	cout << "trans: " << trans << endl;
	cout << "HL: " << HL << endl;
	cout << "Base array extrinsic: " << EXTRINSICCOEFFS[0] << endl;
	cout << "Left array extrinsic: " << EXTRINSICCOEFFS[1] << endl;
	cv::Mat dist1Masked, dist2Masked, dist3Masked, blendMaskSum;

	while (1)
	{
		frameNo++;
		int _startWhileLoop = (int)getTickCount();
		
		std::cout << "Getting Files to stitch..." << std::endl;

		if (stitchFromMemory && !stitchVideo){
			MM->readFrames(FRAMES, camPicPrefix);
		}
		else {
			CO->CO_captureFrames(FRAMES);
		}
		/*imshow("frame1", FRAMES[0]);
		imshow("frame2", FRAMES[1]);
		imshow("frame3", FRAMES[2]);
		cv::waitKey(0);*/

		if (FRAMES[0].empty())
			break;
		std::cout << "Converting to color..." << std::endl;
		IO->IO_cvtColor(FRAMES, CV_RGB2GRAY);
		//IO->IO_resize(FRAMES, cv::Size(frameWidth, frameHeight));
		//IO->IO_transpose(FRAMES);
		//IO->IO_flip(FRAMES, 1);
		/*imshow("frame1", FRAMES[0]);
		imshow("frame2", FRAMES[1]);
		imshow("frame3", FRAMES[2]);
		cv::waitKey(0);*/

		std::cout << "Undistorting..." << std::endl;
		IO->IO_undistort(FRAMES, INTRINSICCOEFFS, DISTORTIONCOEFFS);
		/*imshow("frame1", FRAMES[0]);
		imshow("frame2", FRAMES[1]);
		imshow("frame3", FRAMES[2]);
		cv::waitKey(0);*/

		std::cout << "Equirectangularly warping..." << std::endl;
		IO->IO_rectilinearProject(FRAMES, 0, FOCAL);
		/*imshow("frame1", FRAMES[0]);
		imshow("frame2", FRAMES[1]);
		imshow("frame3", FRAMES[2]);
		cv::waitKey(0);*/

		if (GPU){
			GO->GO_uploadStream(FRAMES);
			GO->GO_warpPerspective(EXTRINSICCOEFFS, resultHeight + 1000, resultWidth);
			GO->GO_downloadStream(RESULTS);
		}
		else {
			std::cout << "Extrapolating pixel position..." << std::endl;
			IO->IO_warpPerspective(FRAMES, RESULTS, EXTRINSICCOEFFS, cv::Size(resultHeight + 1000, resultWidth));
		}
		//MM->writeStaticFrames(RESULTS, 1, "WarpPerspective()");
		/*imshow("frame1", FRAMES[0]);
		imshow("frame2", FRAMES[1]);
		imshow("frame3", FRAMES[2]);
		cv::waitKey(0);*/

		std::cout << "Blending..." << std::endl;
		if (frameNo == 1){

			//BLENDING TEST
			/*imshow("0 - PreBlend", RESULTS[0]);
			imshow("1 - PreBlend", RESULTS[1]);
			waitKey(0);
			*/
			Mat m1, m2, m3;
			//m1 = RESULTS[0](Rect(0, 0, BO->limitPt.rightXLimit, RESULTS[0].rows));
			//m1.setTo(1);
			//m2 = RESULTS[1](Rect(BO->limitPt.leftXLimit, 0, RESULTS[0].cols - BO->limitPt.leftXLimit, RESULTS[0].rows));
			//m2.setTo(1);
			cv::threshold(RESULTS[0], m1, 0, 255, cv::THRESH_BINARY);
			cv::threshold(RESULTS[1], m2, 0, 255, cv::THRESH_BINARY);
			cv::threshold(RESULTS[2], m3, 0, 255, cv::THRESH_BINARY);

			//imshow("0 - Mask", m1);
			//imshow("1 - Mask", m2);
			//imshow("2 - Mask", m3);
			//waitKey(0);

			cv::Mat bothMasks = m1 | m2 | m3;
			//cv::Mat ampersandMasks = m1 & m2 & m3;
			cv::Mat noMask = 255 - bothMasks;

			//imshow("0 - Both Mask", bothMasks);
			//imshow("0 - Ampersand Mask", ampersandMasks);
			//imshow("0 - No Mask", noMask);
			//waitKey(0);

			cv::Mat rawAlpha = cv::Mat(noMask.rows, noMask.cols, CV_32FC1);
			rawAlpha = 1.0f;

			cv::Mat border1 = 255 - border(m1);
			cv::Mat border2 = 255 - border(m2);
			cv::Mat border3 = 255 - border(m3);

			//cv::imshow("0 - border1", border1);
			//cv::imshow("1 - border2", border2);
			//cv::imshow("2 - border3", border3);
			//waitKey(0);

			cv::Mat dist1, dist2, dist3;
			cv::distanceTransform(border1, dist1, CV_DIST_L2, 3);
			cv::distanceTransform(border2, dist2, CV_DIST_L2, 3);
			cv::distanceTransform(border3, dist3, CV_DIST_L2, 3);
			double min, max;
			cv::Point minLoc, maxLoc;
			cv::minMaxLoc(dist1, &min, &max, &minLoc, &maxLoc, m1&(dist1 > 0));
			dist1 = dist1* 1.0 / max;
			cv::minMaxLoc(dist2, &min, &max, &minLoc, &maxLoc, m2&(dist2 > 0));
			dist2 = dist2* 1.0 / max;
			cv::minMaxLoc(dist3, &min, &max, &minLoc, &maxLoc, m3&(dist3 > 0));
			dist3 = dist3* 1.0 / max;

			/*cv::imshow("0 - Distance", dist1);
			cv::imshow("1 - Distance", dist2);
			waitKey(0);
			*/
			rawAlpha.copyTo(dist1Masked, noMask);
			dist1.copyTo(dist1Masked, m1);
			rawAlpha.copyTo(dist1Masked, m1&(255 - m2 - m3));

			rawAlpha.copyTo(dist2Masked, noMask);
			dist2.copyTo(dist2Masked, m2);
			rawAlpha.copyTo(dist2Masked, m2&(255 - m1 - m3));

			rawAlpha.copyTo(dist3Masked, noMask);
			dist3.copyTo(dist3Masked, m3);
			rawAlpha.copyTo(dist3Masked, m3&(255 - m1 - m2));

			/*cv::imshow("0 - Distance Masked", dist1Masked);
			cv::imshow("1 - Distance Masked", dist2Masked);
			cv::imshow("3 - Distance Masked", dist3Masked);
			waitKey(0);*/

			blendMaskSum = dist1Masked + dist2Masked + dist3Masked;
			//cv::imshow("0 - BlendMaskSum", blendMaskSum);
			//waitKey(0);

		}
		/*imshow("frame1", RESULTS[0]);
		imshow("frame2", RESULTS[1]);
		imshow("frame3", RESULTS[2]);
		cv::waitKey(0);*/

		cv::Mat im1Float, im2Float, im3Float;
		RESULTS[0].convertTo(im1Float, dist1Masked.type());
		RESULTS[1].convertTo(im2Float, dist2Masked.type());
		RESULTS[2].convertTo(im3Float, dist3Masked.type());
		cv::Mat im1Alpha = dist1Masked.mul(im1Float);
		cv::Mat im2Alpha = dist2Masked.mul(im2Float);
		cv::Mat im3Alpha = dist3Masked.mul(im3Float);

		/*cv::imshow("0 - Blended", im1Alpha/255.0);
		cv::imshow("1 - Blended", im2Alpha/255.0);
		cv::imshow("2 - Blended", im3Alpha/ 255.0);
		waitKey(0);*/

		cv::Mat imBlended = (im1Alpha + im2Alpha + im3Alpha) / blendMaskSum;

		/*cv::imshow("0 - Blended and Merged", imBlended / 255.0);
		waitKey(0);
		*/

		//BO->BO_alphaBlend(RESULTS, 0.3, result);
		//result = imBlended;
		imBlended.convertTo(result, CV_8UC1);
		//CHANGE
		/*imshow("result", result);
		cv::waitKey(0);*/
		croppedImage = result(Rect(0, 0, result.cols, result.rows / 2));

		if (croppedImage.channels() == 3) {
			cv::cvtColor(croppedImage, croppedImage, CV_RGB2BGR);
		}

		if (stitchFromMemory){
			std::vector<cv::Mat> ResultFinal(1);
			cv::Mat color_img;
			cv::cvtColor(croppedImage, color_img, CV_GRAY2BGR);
			putText(color_img, "moment in the orb", cvPoint(20, 30),
				FONT_HERSHEY_DUPLEX, 0.8, cvScalar(0, 144, 255), 1, CV_AA);
			ResultFinal[0] = color_img;
			if (!stitchVideo){
				imshow("cropped Result", color_img);
				waitKey(0);
				MM->writeStaticFrames(ResultFinal, 1, preprocess + "/FinalStitchedResult");
				break;
			}
			else {
				MM->writeVideoFrames(ResultFinal);
			}
		}
		else {
			imshow("cropped Result", croppedImage);
		}


		//waitKey(0);
		//std::vector<cv::Mat> FINALARRAY(1);
		//FINALARRAY[0] = croppedImage;
		//MM->writeStaticFrames(FINALARRAY, 1, "FinalFrame");

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
	delete CO;
	delete MM;
	delete IO;
	delete GO;
	delete BO;
	return 1;
}

cv::Mat Test::border(cv::Mat mask)
{
	cv::Mat gx;
	cv::Mat gy;

	cv::Sobel(mask, gx, CV_32F, 1, 0, 3);
	cv::Sobel(mask, gy, CV_32F, 0, 1, 3);

	cv::Mat border;
	cv::magnitude(gx, gy, border);

	return border > 100;
}