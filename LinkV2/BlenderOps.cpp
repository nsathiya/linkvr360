#include "BlenderOps.h"


BlenderOps::BlenderOps()
{
}


BlenderOps::~BlenderOps()
{
}

void BlenderOps::BO_alphaBlend(std::vector<cv::Mat> &RESULTS, float blendAlpha, cv::Mat &BLENDEDRESULT){

	for (int j = 0; j < BLENDEDRESULT.rows; ++j)
		for (int i = 0; i < BLENDEDRESULT.cols; ++i)
		{

			float blendA = blendAlpha;
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

			int blendValue;
			bool cL_0 = false;
			bool cB_0 = false;
			bool cR_0 = false;
			bool c4_0 = false;
			bool c5_0 = false;
			bool c6_0 = false;


			if (j < BLENDEDRESULT.rows && i < limitPt.leftXLimit){
				cL_0 = true;
				cL = RESULTS[1].at<uchar>(j, i);
			}
			if (j < BLENDEDRESULT.rows && i > limitPt.rightXLimit) {
				//cout << "cB is true" << endl;
				cB_0 = true;
				cB = RESULTS[0].at<uchar>(j, i);
			}

			
			if (cL_0 && cB_0)
			{
				// Use combination of base + left
				color = ((1 - blendA)*cL + blendA*cB);
			}
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

			BLENDEDRESULT.at<uchar>(j, i) = color;

		}
}