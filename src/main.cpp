/**
 * jiasheng
 * optical flow avoidance
 */
#include<stdio.h>
#include<math.h>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>

#define NUM_FEATURES_TO_TRACK 100
#define MAX_POSSIBLE_HYPOTENUSE_LENGTH 100
#define MEAN_FLOW_THRESHOLD 100

using namespace std;
using namespace cv;

IplImage *outputframe, *frame1_1C, *eig_image, *temp_image, *swap_image_temp;
int number_of_features, block_size, use_harris;
CvPoint2D32f *frame1_features, *swap_points_temp;
double quality, min_distance, harris_free_k;
//*************************//
CvCapture *capture;
IplImage *frame1, *frame2_1C, *pyramid1, *pyramid2;
CvPoint2D32f *frame2_features;
CvSize optical_flow_window;
int max_num_pyramids, FlowPyrLk_flag;
char *optical_flow_found_feature;
float *optical_flow_feature_error;
//*************************//
int skycount, leftgroundcount, rightgroundcount, collisioncount,
		sumcollisiontime;
CvPoint sumskycount, sumleftgroundcount, sumrightgroundcount, p, q;
CvSize imsize;

int square(int x) {
	return x * x;
}
void features_detect() {/*featires detection algorithm*/
	/*CvPoint pt1 = { frame_size.width / 2, frame_size.height / 2 };
	 CvFont font;
	 cvInitFont( &font, CV_FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 0, 2, CV_AA );
	 cvPutText( outputframe, "INIT", pt1, &font, CV_RGB(255,0,0));*/
	/*=====Query FIRST frame=====*/
	//outputframe already available from the Idle Routine
	cvConvertImage(outputframe, frame1_1C, 0);
	/*by setting number_of_features=0 we prevent feature extraction and tracking code from execution*/
	number_of_features = NUM_FEATURES_TO_TRACK;
	/*====== Shi and Tomasi Feature Tracking!======*/
	/* * "frame1_1C" is the input image. = frame1
	 * "eig_image" and "temp_image" are just workspace for the algorithm.
	 * The first ".01" specifies the minimum quality of the features (based on the eigenvalues).
	 * The second ".01" specifies the minimum Euclidean distance between features.
	 * "NULL" means use the entire input image. You could point to a part of the image.
	 * WHEN THE ALGORITHM RETURNS:
	 * "frame1_features" will contain the feature points.
	 * "number_of_features" will be set to a value <= 400 indicating the number of feature points found.
	 * cvGoodFeaturesToTrack( grey, eig, temp, points[1], &count, quality,
	 min_distance, 0, block_size,use_harris,harris_free_k);*/
	if (number_of_features > 0) {
		cvGoodFeaturesToTrack(frame1_1C, eig_image, temp_image, frame1_features,
				&number_of_features, quality, min_distance, NULL, block_size,
				use_harris, harris_free_k);
	}
}
void features_track() {/*features tracking algorithm*/
	/*=====Query SECOND frame=====*/
//	Grab_IEEE1394();
//frame1->imageData=(char *)cam_capture.capture_buffer;
//	FirewireFrame_to_RGBIplImage((char *) cam_capture.capture_buffer, frame1);
//FirewireFrame_to_BWIplImage((char *)cam_capture.capture_buffer, frame1);
//	Release_IEEE1394();
	frame1 = cvQueryFrame(capture);
	cvConvertImage(frame1, frame2_1C, 0);
	/*====== Pyramidal Lucas Kanade Optical Flow!======= */
	/* * "frame1_1C" is the first frame with the known features.
	 * "frame2_1C" is the second frame where we want to find the first frame’s features.
	 * "pyramid1" and "pyramid2" are workspace for the algorithm.
	 * "frame1_features" are the features from the first frame.
	 * "frame2_features" is the (outputted) locations of those features in the second frame.
	 * "number_of_features" is the number of features in the frame1_features array.
	 * "optical_flow_window" is the size of the window to use to avoid the aperture problem.
	 * "5" is the maximum number of pyramids to use. 0 would be just one level.
	 * "optical_flow_found_feature" is as described above (non-zero iff feature found by the flow).
	 * "optical_flow_feature_error" is as described above (error in the flow for this feature).
	 * "optical_flow_termination_criteria" is as described above (how long the algorithm should look).
	 * "0" means disable enhancements. */
	cvCalcOpticalFlowPyrLK(frame1_1C, frame2_1C, pyramid1, pyramid2,
			frame1_features, frame2_features, number_of_features,
			optical_flow_window, max_num_pyramids, optical_flow_found_feature,
			optical_flow_feature_error,
			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3),
			FlowPyrLk_flag);
//REUSE MODE Should be activated after second run of the algorithm
	FlowPyrLk_flag = CV_LKFLOW_PYR_A_READY;
	/*Swap buffers for reuse*/
	CV_SWAP( frame1_1C, frame2_1C, swap_image_temp);
	CV_SWAP( pyramid1, pyramid2, swap_image_temp);
	CV_SWAP( frame1_features, frame2_features, swap_points_temp);
	int i, k;
	for (i = k = 0; i < number_of_features; i++) { /* If Pyramidal Lucas Kanade didn’t really find the feature, skip it. */
		if (optical_flow_found_feature[i] == 0)
			continue;
		/*filter out lost features*/
		frame1_features[k] = frame1_features[i];
		frame2_features[k] = frame2_features[i];
		k++;
	}
	/*set proper number of the features*/
	number_of_features = k;
}
void draw_calc_avg_flow_vectors() {/*draw opical flow vectors*/
	/*Reset counts and sum*/
	skycount = 0;
	sumskycount.x = 0;
	sumskycount.y = 0;
	leftgroundcount = 0;
	rightgroundcount = 0;
	sumleftgroundcount.x = 0;
	sumrightgroundcount.x = 0;
	sumleftgroundcount.y = 0;
	sumrightgroundcount.y = 0;
	collisioncount = 0;
	sumcollisiontime = 0;
	/*===Features filtering===*/
	/*temp variables*/
	int countflow = 0;
	double sumflow = 0;
	double meanflow = 0;
	int tmpcount = 0;
	/*Filter enormous hypotenuse*/
	for (int i = 0; i < number_of_features; i++) {
		p.x = (int) frame1_features[i].x;
		p.y = (int) frame1_features[i].y;
		q.x = (int) frame2_features[i].x;
		q.y = (int) frame2_features[i].y;
		double hypotenuse;
		hypotenuse = sqrt(square(p.y - q.y) + square(p.x - q.x));
		if (hypotenuse < MAX_POSSIBLE_HYPOTENUSE_LENGTH) {
			/*draw unstable features with yellow*/
			cvCircle(outputframe, cvPointFrom32f(frame1_features[i]), 2,
					CV_RGB(255,255,0), -1, 8, 0);
			frame1_features[tmpcount].x = frame1_features[i].x;
			frame1_features[tmpcount].y = frame1_features[i].y;
			frame2_features[tmpcount].x = frame2_features[i].x;
			frame2_features[tmpcount].y = frame2_features[i].y;
			tmpcount++;
		}
	}
	number_of_features = tmpcount;
	/*Estimate mean flow*/
	for (int i = 0; i < number_of_features; i++) {
		p.x = (int) frame1_features[i].x;
		p.y = (int) frame1_features[i].y;
		q.x = (int) frame2_features[i].x;
		q.y = (int) frame2_features[i].y;
		double hypotenuse;
		hypotenuse = sqrt(square(p.y - q.y) + square(p.x - q.x));
		countflow++;
		sumflow += hypotenuse;
	}
	meanflow = sumflow / countflow;
	/*Filter features using mean flow*/
	tmpcount = 0;
	for (int i = 0; i < number_of_features; i++)/*look through all features if number of features > 0*/
	{
		p.x = (int) frame1_features[i].x;
		p.y = (int) frame1_features[i].y;
		q.x = (int) frame2_features[i].x;
		q.y = (int) frame2_features[i].y;
		double hypotenuse;
		hypotenuse = sqrt(square(p.y - q.y) + square(p.x - q.x));
		if ((hypotenuse / meanflow) < MEAN_FLOW_THRESHOLD) //pass features with mean hypotenuse
		{
			frame1_features[tmpcount].x = frame1_features[i].x;
			frame1_features[tmpcount].y = frame1_features[i].y;
			frame2_features[tmpcount].x = frame2_features[i].x;
			frame2_features[tmpcount].y = frame2_features[i].y;
			tmpcount++;
		}
	}
	number_of_features = tmpcount;
	/*Count flow in screen zones*/
	for (int i = 0; i < number_of_features; i++)/*look through all features if number of features > 0*/
	{
		p.x = (int) frame1_features[i].x;
		p.y = (int) frame1_features[i].y;
		q.x = (int) frame2_features[i].x;
		q.y = (int) frame2_features[i].y;
		if (p.y < (imsize.height * 2 / 7)) // top part of the screen to estimate for rotation
				{
			skycount++;
			sumskycount.x += (p.x - q.x);
			sumskycount.y += (p.y - q.y);
		}
		if ((p.y > (imsize.height * 2 / 7)) && (p.x < (imsize.width / 2))) //for left-part of the screen
				{
			leftgroundcount++;
			sumleftgroundcount.x += (p.x - q.x);
			sumleftgroundcount.y += (p.y - q.y);
		}
		if ((p.y > (imsize.height * 2 / 7)) && (p.x > (imsize.width / 2))) //for right-part of the screen
				{
			rightgroundcount++;
			sumrightgroundcount.x += (p.x - q.x);
			sumrightgroundcount.y += (p.y - q.y);
		}
		//for central square time to collision
		if ((p.y > (imsize.height * 2 / 7)) && (p.x > (imsize.width * 2 / 7))
				&& (p.y < (imsize.height * 5 / 7))
				&& (p.x < (imsize.width * 5 / 7))) {
			collisioncount++;
			// Calc inverse collision time
			sumcollisiontime += (p.y - q.y);
		}
	}/*end of looking through all features*/
	/*Draw flow vectors*/
	for (int i = 0; i < number_of_features; i++) {
		p.x = (int) frame1_features[i].x;
		p.y = (int) frame1_features[i].y;
		q.x = (int) frame2_features[i].x;
		q.y = (int) frame2_features[i].y;
		double angle;
		angle = atan2((double) p.y - q.y, (double) p.x - q.x);
		double hypotenuse;
		hypotenuse = sqrt(square(p.y - q.y) + square(p.x - q.x));
		/*-------draw flow vectors-------*/
		cvCircle(outputframe, cvPointFrom32f(frame1_features[i]), 2,
				CV_RGB(0,255,0), -1, 8, 0);
		/* Here we lengthen the arrow by a factor of 5. */
		q.x = (int) (p.x - 2 * hypotenuse * cos(angle));
		q.y = (int) (p.y - 2 * hypotenuse * sin(angle));
		/* Now we draw the main line of the arrow.
		 * "p" is the point where the line begins.
		 * "q" is the point where the line stops. * "CV_AA" means antialiased drawing.
		 * "0" means no fractional bits in the center cooridinate or radius.*/
		cvLine(outputframe, p, q, CV_RGB(200,0,0), 1, CV_AA, 0);
		/* Now draw the tips of the arrow. I do some scaling so that the
		 * tips look proportional to the main line of the arrow.*/
		p.x = (int) (q.x + 4 * cos(angle + M_PI / 4));
		p.y = (int) (q.y + 4 * sin(angle + M_PI / 4));
		cvLine(outputframe, p, q, CV_RGB(200,0,0), 1, CV_AA, 0);
		p.x = (int) (q.x + 4 * cos(angle - M_PI / 4));
		p.y = (int) (q.y + 4 * sin(angle - M_PI / 4));
		cvLine(outputframe, p, q, CV_RGB(200,0,0), 1, CV_AA, 0);
	}
}
int main() {
	puts("hello world");
	capture = cvCaptureFromCAM(0);
	outputframe = cvQueryFrame(capture);
	return 1;
}
