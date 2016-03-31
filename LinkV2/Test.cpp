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

cv::Mat _pic1, _pic2, _pic3, _picResult;

cv::Mat Test::getWorld(cv::Mat pic1, cv::Mat pic2, cv::Mat pic3){

	//cv::imwrite("testPicturesaveFromc++1.jpg", pic1);
	//cv::imwrite("testPicturesaveFromc++2.jpg", pic2);
	//cv::imwrite("testPicturesaveFromc++3.jpg", pic3);
	
	_pic1 = pic1;
	_pic2 = pic2;
	_pic3 = pic3;
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
			return _picResult; //errno;
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
		testingFunction2(true, true, false);
	}
	
	return _picResult;
}

const int BASE_CAM = 1;  //0;
const int LEFT_CAM = 0;  //2;  //1; // 3;
const int RIGHT_CAM = 2; // 1; // 2; // 4;
const int FOUR_CAM = 1;
const int FIFTH_CAM = 2;
const int BACK_CAM = 5;
const int NO_OF_CAMS = 3;

std::string videoPath = "../LinkV2/samples"; //"samples"; 
std::string preprocess = videoPath + "/preprocess";
std::string calibrationPath = "../LinkV2/calibration"; //"calibration";
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
bool useGrayScale = false;
cv::Mat border(cv::Mat mask);
std::vector<cv::Mat> FRAMES(NO_OF_CAMS);
std::vector<cv::Mat> INTRINSICCOEFFS(NO_OF_CAMS);
std::vector<cv::Mat> EXTRINSICCOEFFS(NO_OF_CAMS);
std::vector<cv::Mat> DISTORTIONCOEFFS(NO_OF_CAMS);
std::vector<cv::Mat> RESULTS(NO_OF_CAMS);
std::vector<cv::Mat> THRESHOLDED(NO_OF_CAMS);
std::vector<std::string> PREPROCESS_FRAMES_PATH(NO_OF_CAMS);
std::vector<std::string> RESULTS_FRAMES_PATH(1);
std::vector<float> FOCAL(NO_OF_CAMS);

void Test::setup()
{

	FOCAL[0] = CAM_F_MAP[BASE_CAM] = 3.18; //395.164;
	FOCAL[1] = CAM_F_MAP[LEFT_CAM] = 3.2; //422.400;
	FOCAL[2] = CAM_F_MAP[RIGHT_CAM] = 3.16; //326.38;
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
			//MM->readFrames(FRAMES, camPicPrefix);
			FRAMES[0] = _pic1; 
			FRAMES[1] = _pic2;
			FRAMES[2] = _pic3;
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
		std::cout << "Converting color..." << std::endl;
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
		std::cout << "Finished setting up blend template" << std::endl;
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
		std::cout << "FInsihed blending" << std::endl;
		croppedImage = result(Rect(0, 0, result.cols, result.rows / 2));

		std::cout << "finished cropping" << std::endl;
		if (croppedImage.channels() == 3) {
			cv::cvtColor(croppedImage, croppedImage, CV_RGB2BGR);
		}

		_picResult = croppedImage;

		if (stitchFromMemory){
			std::vector<cv::Mat> ResultFinal(1);
			cv::Mat color_img;
			cv::cvtColor(croppedImage, color_img, CV_GRAY2BGR);
			putText(color_img, "moment in the orb", cvPoint(20, 30),
				FONT_HERSHEY_DUPLEX, 0.8, cvScalar(0, 144, 255), 1, CV_AA);
			ResultFinal[0] = color_img;
			if (!stitchVideo){
				//imshow("cropped Result", color_img);
				//waitKey(0);
				std::cout << "saving pciture" << std::endl;
				MM->writeStaticFrames(ResultFinal, 1, preprocess + "/FinalStitchedResult");
				std::cout << "saved" << std::endl;
				break;
			}
			else {
				MM->writeVideoFrames(ResultFinal);
			}
		}
		else {
			std::cout << "Trying to show cropped image" << std::endl;
			imshow("cropped Result", croppedImage);
			std::cout << "image shown" << std::endl;
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
		std::cout << "calculated time info" << std::endl;

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
	std::cout << "returning" << std::endl;
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


////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int Test::testingFunction2(bool GPU, bool stitchFromMemory, bool stitchVideo) {

	std::string method = "w/o GPU";
	if (GPU)
		method = "w GPU";

	cout << "Stitching " + method << endl;
	cout << "Stitching from memory: " << stitchFromMemory << endl;
	if (stitchFromMemory)
		cout << "If stitch from memory, stitch video: " << stitchVideo << endl;

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
	CO->CO_setProp(CV_CAP_PROP_FRAME_WIDTH, 1640);
	CO->CO_setProp(CV_CAP_PROP_FRAME_HEIGHT, 1080);


	int frameWidth = CO->CO_getProp(CV_CAP_PROP_FRAME_WIDTH, 0); // *0.25;
	int frameHeight = CO->CO_getProp(CV_CAP_PROP_FRAME_HEIGHT, 0); // *0.25;
	int resultWidth = frameWidth * 2;
	int resultHeight = frameHeight + 100;
	bool record = false;
	cout << frameWidth << " " << frameHeight << endl;
	cv::Mat result = Mat(resultWidth, resultHeight, useGrayScale ? CV_16UC1 : CV_16UC3);
	if (stitchFromMemory && !stitchVideo)
		MM = new MemoryManager(NO_OF_CAMS, PREPROCESS_FRAMES_PATH, frameHeight, frameWidth, stitchVideo);
	if (stitchFromMemory && stitchVideo)
		MM = new MemoryManager(1, RESULTS_FRAMES_PATH, resultHeight, resultWidth/2, stitchVideo);
	
	// Move Scene to the right by 100
	int x_offset = 400.0;
	float y_offset = 100.0;
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

	//IO->IO_resize(FRAMES, cv::Size(frameWidth, frameHeight));
	//IO->IO_transpose(FRAMES);
	//IO->IO_flip(FRAMES, 1);
	//IO->IO_undistort(FRAMES, INTRINSICCOEFFS, DISTORTIONCOEFFS);

	
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

	perspectiveTransform(scene_cornersBase, scene_cornersBase, trans);
	perspectiveTransform(scene_cornersLeft, scene_cornersLeft, HL);
	perspectiveTransform(scene_cornersRight, scene_cornersRight, HR);

	//Store useful information for Image limits
	int leftLimit, baseLeftLimit, baseRightLimit, rightLimit, fourLimit, fifthLimit, sixLimit;

	//fifthLimit = scene_cornersFive[1].x;
	////sixLimit = scene_cornersSix[1].x;
	leftLimit = scene_cornersLeft[1].x;
	baseLeftLimit = x_offset;
	//baseRightLimit = x_offset + FRAMES[0].cols;
	//rightLimit = scene_cornersRight[0].x;
	//fourLimit = scene_cornersFour[0].x;
	Mat croppedImage;
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
	RESULTS[0] = cv::Mat(resultWidth, resultHeight, CV_8UC3);
	RESULTS[1] = cv::Mat(resultWidth, resultHeight, CV_8UC3);
	RESULTS[2] = cv::Mat(resultWidth, resultHeight, CV_8UC3);
	THRESHOLDED[0] = cv::Mat(resultWidth, resultHeight, CV_8UC3);
	THRESHOLDED[1] = cv::Mat(resultWidth, resultHeight, CV_8UC3);
	THRESHOLDED[2] = cv::Mat(resultWidth, resultHeight, CV_8UC3);


	//RESULTS[3] = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	//RESULTS[4] = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	//outSixFrame = cv::Mat(resultWidth, resultHeight + 600, useGrayScale ? CV_8UC1 : CV_8UC3);
	//Start processing
	cout << "trans: " << trans << endl;
	cout << "HL: " << HL << endl;
	cout << "HR: " << HR << endl;
	
	cout << "Base array extrinsic: " << EXTRINSICCOEFFS[0] << endl;
	cout << "Left array extrinsic: " << EXTRINSICCOEFFS[1] << endl;
	cout << "right array extrinsic: " << EXTRINSICCOEFFS[2] << endl;
	
	cv::Mat dist1Masked, dist2Masked, dist3Masked, blendMaskSum;

	while (1)
	{
		frameNo++;
		int _startWhileLoop = (int)getTickCount();

		if (stitchFromMemory && !stitchVideo){
			FRAMES[0] = _pic1; 
			FRAMES[1] = _pic2;
			FRAMES[2] = _pic3;
		}
		else {
			cout << "capturing frames" << endl;
			CO->CO_captureFrames(FRAMES);
		}
		//imshow("base frame", FRAMES[0]);
		//imshow("left frame", FRAMES[1]);
		//imshow("right frame", FRAMES[2]);

		cv::imwrite("baseFrame.jpg", FRAMES[0]);
		cv::imwrite("leftFrame.jpg", FRAMES[1]);
		cv::imwrite("rightFrame.jpg", FRAMES[2]);

		//cv::waitKey(0);
		if (FRAMES[0].empty())
			break;
		
		cv::Matx33d newK1 = INTRINSICCOEFFS[0];
		newK1(0, 0) = 200;
		newK1(1, 1) = 300;
		cv::Matx33d newK2 = INTRINSICCOEFFS[1];
		newK2(0, 0) = 200;
		newK2(1, 1) = 300;

		cv::resize(FRAMES[0], FRAMES[0], cv::Size(900, 500));
		cv::resize(FRAMES[1], FRAMES[1], cv::Size(900, 500));
		cv::resize(FRAMES[2], FRAMES[2], cv::Size(900, 500));
		cv::Rect myROI(50, 0, 800, 500), myROI2(150, 150, 500, 500);

		cv::Mat row1 = cv::Mat::ones(150, 800, FRAMES[0].type());  // 3 cols
		cv::Mat row2 = cv::Mat::ones(150, 800, FRAMES[0].type());  // 3 cols
		cv::Mat croppedImageB, croppedImageL, croppedImageR;

		croppedImageB.push_back(row1);
		croppedImageL.push_back(row1);
		croppedImageR.push_back(row1);
		croppedImageB.push_back(FRAMES[0](myROI));
		croppedImageL.push_back(FRAMES[1](myROI));
		croppedImageR.push_back(FRAMES[2](myROI));
		croppedImageB.push_back(row1);
		croppedImageL.push_back(row1);
		croppedImageR.push_back(row1);

		cv::Mat rectLinearBaseFrame = rectlinearProject(croppedImageB, 0, FOCAL[0]);
		cv::Mat rectLinearLeftFrame = rectlinearProject(croppedImageL, 0, FOCAL[1]);
		cv::Mat rectLinearRightFrame = rectlinearProject(croppedImageR, 0, FOCAL[2]);

		rectLinearBaseFrame = rectLinearBaseFrame(myROI2);
		rectLinearLeftFrame = rectLinearLeftFrame(myROI2);
		rectLinearRightFrame = rectLinearRightFrame(myROI2);

		//imshow("base Frame", rectLinearBaseFrame);
		//imshow("left Frame", rectLinearRightFrame);

		FRAMES[0] = rectLinearBaseFrame;
		FRAMES[1] = rectLinearLeftFrame;
		FRAMES[2] = rectLinearRightFrame;

		//imshow("base frame", FRAMES[0]);
		//imshow("left frame", FRAMES[1]);
		//imshow("right frame", FRAMES[2]);
		//imwrite("baseframe-abouttowarp.jpg", FRAMES[0]);
		//imwrite("leftframe-abouttowarp.jpg", FRAMES[1]);
		//imwrite("rightframe-abouttowarp.jpg", FRAMES[2]);

		//cv::waitKey(0);
		//IO->IO_rectilinearProject(FRAMES, 0, FOCAL);
		if (false){
			GO->GO_uploadStream(FRAMES);
			GO->GO_warpPerspective(EXTRINSICCOEFFS, resultHeight, resultWidth);
			GO->GO_downloadStream(RESULTS);
		}
		else {
			IO->IO_warpPerspective(FRAMES, RESULTS, EXTRINSICCOEFFS, cv::Size(resultHeight, resultWidth));
		}
		//MM->writeStaticFrames(RESULTS, 1, "WarpPerspective()");
		
		//imshow("base frame", RESULTS[0]);
		//imshow("left frame", RESULTS[1]);
		//imshow("right frame", RESULTS[2]);
		//cv::waitKey(0);
		//imwrite("baseframe-warped.jpg", RESULTS[0]);
		//imwrite("leftframe-warped.jpg", RESULTS[1]);
		//imwrite("rightframe-warped.jpg", RESULTS[2]);
		RESULTS[0].copyTo(THRESHOLDED[0]);
		RESULTS[1].copyTo(THRESHOLDED[1]);
		RESULTS[2].copyTo(THRESHOLDED[2]);
		cv::vector<cv::Mat> result1split(4), result2split(4), result3split(4);
		cv::split(RESULTS[0], result1split);
		cv::split(RESULTS[1], result2split);
		cv::split(RESULTS[2], result3split);

			cout << "entering blending" << endl;
			Mat m1, m2, m3;

			IO->IO_cvtColor(THRESHOLDED, CV_BGR2GRAY);
			cv::threshold(THRESHOLDED[0], m1, 0, 255, cv::THRESH_BINARY);
			cv::threshold(THRESHOLDED[1], m2, 0, 255, cv::THRESH_BINARY);
			cv::threshold(THRESHOLDED[2], m3, 0, 255, cv::THRESH_BINARY);
			cout << "thresholded" << endl;
			cv::vector<cv::Mat> threshold1split, threshold2split, threshold3split;
			cv::split(m1, threshold1split);
			cv::split(m2, threshold2split);
			cv::split(m3, threshold3split);

			Mat rgba0(RESULTS[0].rows, RESULTS[0].cols, CV_8UC4, Scalar(1, 2, 3, 4));
			Mat rgba1(RESULTS[1].rows, RESULTS[1].cols, CV_8UC4, Scalar(1, 2, 3, 4));
			Mat rgba2(RESULTS[2].rows, RESULTS[2].cols, CV_8UC4, Scalar(1, 2, 3, 4));

			Mat bgr0(rgba0.rows, rgba0.cols, CV_8UC3);
			Mat bgr1(rgba1.rows, rgba1.cols, CV_8UC3);
			Mat bgr2(rgba2.rows, rgba2.cols, CV_8UC3);

	
			// forming an array of matrices is a quite efficient operation,
			// because the matrix data is not copied, only the headers
			
			Mat in0[] = { RESULTS[0], threshold1split[0] };
			Mat in1[] = { RESULTS[1], threshold2split[0] };
			Mat in2[] = { RESULTS[2], threshold3split[0] };

			// rgba[0] -> bgr[2], rgba[1] -> bgr[1],
			// rgba[2] -> bgr[0], rgba[3] -> alpha[0]
			int from_to[] = { 0, 0, 1, 1, 2, 2, 3, 3 };
			mixChannels(in0, 2, &rgba0, 1, from_to, 4);
			mixChannels(in1, 2, &rgba1, 1, from_to, 4);
			mixChannels(in2, 2, &rgba2, 1, from_to, 4);

			cout << "number of channels are " << rgba0.channels() << endl;

			cv::imwrite("resultBeforeBlend0.png", rgba0);
			cv::imwrite("resultBeforeBlend1.png", rgba1);
			cv::imwrite("resultBeforeBlend2.png", rgba2);

			system("enblend resultBeforeBlend0.png resultBeforeBlend1.png resultBeforeBlend2.png ");
				
		result = cv::imread("a.tif");
		croppedImage = result(Rect(250,150, 800, 400));
		_picResult = croppedImage;

		break;
		if (stitchFromMemory){
			std::vector<cv::Mat> ResultFinal(1);
			cv::Mat color_img;
			cv::cvtColor(croppedImage, color_img, CV_GRAY2BGR);
			putText(color_img , "moment in the orb", cvPoint(20, 30),
				FONT_HERSHEY_DUPLEX, 0.8, cvScalar(0,144,255),1, CV_AA);
			ResultFinal[0] = color_img;
			if (!stitchVideo){
				//imshow("cropped Result", color_img);
				//waitKey(0);
				//MM->writeStaticFrames(ResultFinal, 1, preprocess + "/FinalStitchedResult");
				break;
			}
			else {
				MM->writeVideoFrames(ResultFinal);
			}
		}
		else {
			imshow("cropped Result", croppedImage);
		}
			
	}
	delete CO;
	delete MM;
	delete IO;
	delete GO;
	delete BO;
	return 1;
}

cv::Mat Test::rectlinearProject(Mat ImgToCalibrate, bool INV_FLAG, float F)
{
	Mat Img = ImgToCalibrate;
	int height = Img.rows;
	int width = Img.cols;
	Mat destPic = Mat(cv::Size(width, height), ImgToCalibrate.type(), cv::Scalar(0,0,0));

	cout << width << height << endl;
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
				destPic.at<uchar>(y, x) = Img.at<uchar>(top_left.y, top_left.x);
				//interpolateBilinear(Img, current_pos, top_left, destPic.at<uchar>(y, x));
			}
			else {//JH: added color pixels
				destPic.at<cv::Vec3b>(y, x) = Img.at<cv::Vec3b>(top_left.y, top_left.x);
				//interpolateBilinear(Img, current_pos, top_left, destPic.at<cv::Vec3b>(y, x));
			}

		}
	}
	return destPic;
}


cv::Point2f Test::convert_pt(cv::Point2f point, int w, int h, int INV_FLAG, float F)
{

	//center the point at 0,0
	float FOV = F; //3.2;
	// Polar angles
	//cv::Point2f pc(point.x - w / 2, point.y - h / 2);
	float theta = 2.0 * 3.14159265 * (point.x / w - 0.5); // -pi to pi
	float phi = 3.14159265 * (point.y / h - 0.5);	// -pi/2 to pi/2
	
	// Vector in 3D space
	float x = cos(phi) * sin(theta);
	float y = cos(phi) * cos(theta);
	float z = sin(phi);

	// Calculate fisheye angle and radius
	theta = atan2(z, x);
	phi = atan2(sqrt(x*x + z*z), y);
	float r = w * phi / FOV;
	// Pixel in fisheye space
	cv::Point2f rP;
	rP.x =  0.5 * w + r * cos(theta);
	rP.y =	0.5 * w + r * sin(theta);

	return rP;
}