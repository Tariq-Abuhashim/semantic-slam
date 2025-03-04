
#ifndef ENGINE_H
#define ENGINE_H

#include "MaskRCNN.hpp"
#include "DoN.hpp"
#include "Inventory.hpp"
#include "Object.hpp"
#include "ObjectPoint.hpp"
#include "ObjectDrawer.hpp"
#include "InstanceViewer.hpp"

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <unordered_set>
#include <thread>

#include "Thirdparty/ORB_SLAM2/include/KeyFrame.h"

using namespace ORB_SLAM2;

/** @brief An engine implementation to run object instance segmentation using maskrcnn-benchmark and ORBSLAM2.
	@author Tariq Abuhashim
	@date August 2019
*/
class Engine
{

	public:
	
	/** Constructor, Initialises Mask-RCNN.
	@param categories a map of integers and object class labels of type unordered_map<int,string>
	*/
	Engine(std::unordered_map<int,std::string> categories, std::string);

	/** Default destructor, Deletes Mask-RCNN.
	*/
	~Engine();

	/**	Main function to run the object instance segmentation process
	@param imRGB is the current KeyFrame RGB image
	@param imD is the current KeyFrame depth image
	@param KF pointer to the current KeyFrame
	*/
	void Run(cv::Mat imRGB, cv::Mat imD, KeyFrame* KF);

	/**	Implements object instance tracking in images by reprojecting map points into the current image frame
	@param maskContour a vector of object contours extracted using OpenCV.
	@param label the classification label of the detection object using MaskRCNN.
	@param score the classification score of the detection object using MaskRCNN.
	@param imDepth the current depth image.
	*/
	Object* TrackObjectPoints(Contour maskContour, const std::string label, double score, cv::Mat imDepth);

	/**	Implements object instance tracking in images by tracking its contours and using epipolar geometry
	@param maskContour a vector of object contours extracted using OpenCV.
	@param label the classification label of the detection object using MaskRCNN.
	@param score the classification score of the detection object using MaskRCNN.
	@param imDepth the current depth image.
	*/
	Object* TrackObjectContours(Contour maskContour, const std::string label, double score, cv::Mat imDepth);

	/**	Projects a map point from 3D into 2D image frame
	@param MP a pointer to the current ObjectPoint.
	*/
	cv::Point ProjectIntoCurrentKF(ObjectPoint* MP);

	/**	Computes the fundamental matrix between two ORBSLAM2 KeyFrames (projects a 2D point in KF1 to a line in KF2)
	@param KF1 a pointer to the first (reference) KeyFrame.
	@param KF2 a pointer to the second KeyFrame.
	*/
	cv::Mat ComputeFundamental(KeyFrame* KF1,  KeyFrame* KF2);
	
	/**	Generates skew-symmetric square matrix (3x3) from an input vector
	@param v input vector with dimensions (3x1)
	*/
	cv::Mat GetSkewSymmetricMatrix(const cv::Mat& v);

	/**	Calculates the shorted distance between a 2D point and a 2D line 
	(  a*x + b*y + c = 0  )
	@param u horizontal coordinate of the 2D point in the image
	@param v vertical coordinate of the 2D point in the image
	@param a first line coefficient (first component of the normal vector to the line)
	@param b second line coefficient (second component of the normal vector to the line)
	@param c third line coefficient
	*/
	float shortest_distance(float u, float v, float a, float b, float c);

	/**	Displays object measurements in 2D on the image. This includes object contours and 3D points projections in 	the current KeyFrame
	@param O a pointer to the object
	@param imRGB the current RGB image
	@param KF the current KeyFrame
	@param maskContour a vector of visible object contours in the current KeyFrame
	@param bbox an OpenCV rectangular coordinates of the current object bounding box
	@param hierarchy OpenCV contours hierarchy
	@param color display color for both, contours and 2D ObjectPoints
	*/
	void display(Object* O, cv::Mat imRGB, Contour maskContour, cv::Rect bbox, std::vector<cv::Vec4i> hierarchy, 							cv::Scalar color);

	bool IsInCurrentKF(float range, cv::Point cvpoint, int margin);

	private:

	std::vector<cv::Point> fuse_segments(map<int, std::vector<cv::Point> >& geometry, Contour texture);

	protected:

	int mSensor;
	/// Minimum perceived depth
	float mnMinDepth;
	/// Maximum perceived depth
	//float mnMaxDepth=3.0; // 3.0 for RGBD
	float mnMaxDepth; // 30.0 Lidar
	/// Threshold for point-to-polygon test
	float mnDist;
	float mMinArea;
	float mMaxArea;
	float mOverlap;
	int mMinPointCount;
	float mProbThd;
	float mRes;

	/// Camera intrinsic parameters (TUM)
	//Mat K = (Mat_<float>(3,3) << 535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1);
	/// Camera intrinsic parameters
	cv::Mat K;
	/// Camera lense distortion parameters
	cv::Mat DistCoef;

	int mWidth, mHeight;	
	KeyFrame* mpCurrentKF;

	/// A pointer to maskrcnn object
	MaskRCNN * mpMaskRCNN;
	/// A pointer to DoN object
	DoN * mpDoN;
	/// Labelled object categories
	std::unordered_map<int,std::string> mmCategories;
	/// A pointer to Inventory
	Inventory * mpInventory;
	/// A pointer to Object viewer
	InstanceViewer* mpViewer;
	/// A pointer to Object drawer
	ObjectDrawer* mpObjectDrawer;
	/// A pointer to Object viewer thread
	std::thread* mptViewer;

	#ifdef COMPILEDWITHC11
	typedef std::chrono::steady_clock::time_point timer;
	#else
	typedef std::chrono::monotonic_clock::time_point timer;
	#endif


	protected:

	timer mtimer;
	bool timeIsTicked = false;
	void tick();
	double tock();

	template <class T>
	T tick() 
	{
		#ifdef COMPILEDWITHC11
		return std::chrono::steady_clock::now();
		#else
		return std::chrono::monotonic_clock::now();
		#endif
	}

	template <class T>
	double tock(T t1) 
	{
		#ifdef COMPILEDWITHC11
		T t2 = std::chrono::steady_clock::now();
		#else
		T t2 = std::chrono::monotonic_clock::now();
		#endif
		return std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
	}

};

#endif // ENGINE_H
