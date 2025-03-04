
#ifndef OBJECT_H
#define OBJECT_H

#include <vector>
#include <map>
#include <set>
#include "Thirdparty/ORB_SLAM2/include/KeyFrame.h"
#include "cv.h"
#include "ObjectPoint.hpp"
#include "tsdf.hpp"
#include "TSDFfusion.hpp"

#include <mutex>

using namespace ORB_SLAM2;

typedef std::vector<std::vector<cv::Point> > Contour;

/** @brief A definition of an object in the inventory
	@author Tariq Abuhashim
	@date August 2019
*/
class Object
{
	public:

		/**	Object constructor
		@param h height of the object depth image (called to initialise an object and its TSDF)
		@param w width of the object depth image (called to initialise an object and its TSDF)
		*/
		Object(KeyFrame* pKF, cv::Rect Box, cv::Mat imD,cv::Mat K, int Sensor);

		/** Object destructor (called to delete an object TSDF from memory)
		*/
		~Object();

		/** Custom compare function to sort map objects using their index (mnId)
		*/
		static bool lId(Object* pMO1, Object* pMO2){
        	return pMO1->mnId<pMO2->mnId;
    	}

		void SetObjectParams( float ProbThd, float MinObjectPoints, int Height, int Width, 
			float MinDepth, float MaxDepth, float Dist, float Res);

		/** A function that includes a KeyFrame into the set of KeyFrames where the object is visible.
		@param pKF the current KeyFrame to add
		*/
		void AddKeyFrame(KeyFrame* pKF);

		/** A function that includes a ObjectPoint into the set of ObjectPoints that belongs to an object is the map.
		Hence, this pointer also links the object to 
		@param pMP the current ObjectPoint to add (mvKeys) in its (KF) - Refer to ORBSLAM2::KeyFrame for details
		*/
		void AddObjectPoint(ObjectPoint* pMP);

		/** Returns a vector of all KeyFrame pointers with an the object contour observations
		*/
		vector<KeyFrame*> GetAllKeyFrames();

		/** Returns a vector of all ObjectPoint pointers of the object in the map
		*/
    	vector<ObjectPoint*> GetAllObjectPoints();

		/** returns a pointer to the map object reference KeyFrame (usually the first KeyFrame seeing object)
		*/
		KeyFrame* GetReferenceKeyFrame();

		/** Returns a map of Keyframes observing the contour and associated index in keyframe
		*/
    	map<KeyFrame*, size_t> GetObservations(); 

		/** Returns the number of contours of the objects in the dataset
		*/
    	int Observations();

		/** Adds details of a contour obsevation into the object. 
		This function is used to know which contour in mvpContours is in KF. 
		The implementation allows a contour to be called by either its KeyFrame or index.
		@param pKF the current KeyFrame where the contour is
		@param idx the index of the countour in the object.
		*/
		void AddObservation(KeyFrame* pKF, size_t idx);

		/** Removes a contour obsevation into the object.
		This function is used to delete the contour with index (idx) which is visible in KeyFrame (KF).
		The contour is deleted from mvpContours.
		@param pKF the current KeyFrame where the contour is
		*/
    	void EraseObservation(KeyFrame* pKF);

		/** Adds a contour obsevation into the object vector of contours (mvpContours)
		@param cnt the current contour
		*/
		void AddContour(const Contour& cnt); // FIXME: this should be in KeyFrame.h

		/** Returns a contour observation by index (idx).
		@param idx is the called contour index
		*/
		Contour GetContour(size_t idx); // FIXME: this should be in KeyFrame.h

		/** Returns a contour observation by KeyFrame (KF). 
		This method is more reliable at linking contours to KeyFrames than using indexing.
		@param KF is a pointer to KeyFrame viewing the object.
		*/
		Contour GetContour(KeyFrame* KF);

		/** Returns the total number of object contours in the dataset
		*/
		size_t GetNumberOfContours(); // FIXME: this should be in KeyFrame.h
		
		/** Adds an object bounding box.
		@param box the current bounding box to be added.
		*/
		void AddBoundingBox(const cv::Rect& box); // FIXME: should be in KeyFrame.cc

		/** Sets the classification label of the object.
		@param label the current label from MaskRCNN or label-fusion rule ?
		*/
		void SetLabel(const std::string& label);

		/** Returns the current classification label of the object.
		*/
		string GetLabel();

		/** Updates the score of the object given most recent contour/MaskRCNN obsevations.
		@param score the current score from MaskRCNN or score-fusion rule ?
		*/
		void UpdateScore(const double score);

		/** Returns the current classification score of the object.
		*/
		double GetScore();
		
		/** Verifies if a ObjectPoint lays inside an object contours.
		@param MP a pointer to the current ObjectPoint to test.
		*/
		bool IsInObject(ObjectPoint* MP);

		/** Verifies if an object contour contains enough 2D map points to be considered a valid object measurement.
		*/
		bool HasEnoughObjectPoints();

		/** Returns the maximum (or total) number of 2D point observations that matches the object contours
		This reflects how many times a ObjectPoint has agreed with this object contours.
		*/
		int GetMaxObservations();

		/** Projects a 3D ObjectPoint into a 2D KeyFrame image
		@param MP a pointer to the current ObjectPoint
		@param KF a pointer to the current KeyFrame
		@param K intrinsic calibration parameters
		*/
		cv::Point ProjectObjectPoint(ObjectPoint* MP, KeyFrame* KF);

		/** Iterates through all map points and add points the agrees with the current contour into the object
		@param maskContour Object contours
		@param imDepth the current KeyFrame depth image
		@param KF pointer to the current ORBSLAM2::KeyFrame
		*/
		void AddNewObjectPoints(Contour maskContour, const cv::Mat& imDepth, KeyFrame* KF, float score); // MaskRCNN
		void AddNewObjectPoints(Contour maskContour, const std::vector<std::vector<cv::Point> >& clusters, 
								const cv::Mat& imDepth, KeyFrame* KF, float score); // MaskRCNN + DoN

		void AddSegment(Contour maskContour, const cv::Mat& imDepth, KeyFrame* KF, float score);  // MaskRCNN
		void AddSegment(const std::vector<cv::Point>& cluster, const cv::Mat& imDepth, 
								KeyFrame* KF, float score);  // MaskRCNN + DoN

		/** Iterates through all object points and verifies the good and bad ones, then deletes/flags the bad ones.
		@param KF2 pointer to the current ORBSLAM2::KeyFrame
		*/
		void CheckObjectPoints(KeyFrame* KF2);
		
		/** Integrates the current depth image into the object TSDF
		*/
		void Integrate(cv::Mat imRGB, cv::Mat imD, cv::Mat T);

		/** Sets the object bad (FIXME: not implemented yet)
		*/
		void SetBadFlag();

		/** Returns true if the object is bad (FIXME: not implemented yet)
		*/
    	bool isBad();

		std::vector<cv::Point> Get2dFeatures(KeyFrame* KF, const cv::Mat& imDepth);
		void SaveToFile(const std::string& ext);

	protected:
		
		float RandomFloat();

		/// A pointer to protected set of KeyFrame seeing this object
		std::set<KeyFrame*> mspKeyFrames; 
		/// A pointer to protected set of ObjectPoint belonging to this object
    	std::set<ObjectPoint*> mspObjectPoints; 
		/// Reference KeyFrame
		KeyFrame* mpRefKF;
		/// A vector of protected object contours
		std::vector<Contour> mvpContours; // FIXME: this should be in KeyFrame.h and is linked to using mObservations
		/// A vector of protected object bounding boxes
		std::vector<cv::Rect> mvpBoundingBoxes;
		/// A vector of protected object segments
		std::vector<std::vector<cv::Point> > mvpSegments;
		/// An object label (string)
		std::string msLabel;
		/// An object classification score (double)
		double mnScore;
		/// Keyframes observing the contour and associated index in keyframe->mvpContours
     	std::map<KeyFrame*, size_t> mObservations;
		/// Bad flag (we do not currently erase ObjectPoint from memory)
     	bool mbBad;
		/// Minimum number of ObjectPoint an object should allowed to have inorder to be included into the inventory
		size_t mdMinPointsCount;
		/// A pointer to an object TSDF
		TSDF* tsdf; // cpp
		//TSDFfusion* tsdf; // python

	protected:

		/// An object depth image heights
		int mnHeight;
		/// An object depth image width
		int mnWidth;
		/// Minimum perceived depth
		float mnMinDepth;
		/// Maximum perceived depth
		float mnMaxDepth;
		/// Object resolution
		float mnRes;
		/// Point-to-polygon test
		float mnDist;

		float mProbThd;
		int mSensor;

		/// Camera intrinsic parameters (TUM)
		//cv::Mat K = (cv::Mat_<float>(3,3) << 535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1); // RGBD
		/// Camera intrinsic parameters (Kitti)
		cv::Mat mK;// = (cv::Mat_<float>(3,3) << 721.5377, 0, 609.5593, 0, 721.5377, 172.854, 0, 0, 1); // lidar
		/// Camera lense distortion parameters
		cv::Mat mDistCoef = (cv::Mat_<float>(4,1) << 0, 0, 0, 0);

		std::mutex mMutexObject;

	public:
		
		/// Current Object Id
		long unsigned int mnId;
		/// Next Object Id
		static long unsigned int nNextId;
		/// number of Countours in the object
		int nObs;
		/// number of ObjectPoints in the object
		int mnTracks;

		float mRed, mGreen, mBlue;

};

#endif // OBJECT_H
