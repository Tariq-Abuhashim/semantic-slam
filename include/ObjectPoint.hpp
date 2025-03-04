#ifndef OBJECTPOINT_H
#define OBJECTPOINT_H

#include "Thirdparty/ORB_SLAM2/include/KeyFrame.h"

#include "cv.h"
#include <mutex>

using namespace ORB_SLAM2;

class ObjectPoint
{
public:
    ObjectPoint(const cv::Mat &Pos, KeyFrame* pRefKF);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(ObjectPoint* pMP);    
    ObjectPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

	void UpdateForegroundProbability(float score);
	void UpdateBackgroundProbability(float score);

	float GetForegroundProbability();
	float GetBackgroundProbability();

	float GetPointProbability();
	void SetProbabilityThreshold(float mProbThd);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;

protected:    

	// Position in absolute coordinates
	cv::Mat mWorldPos;

	// Keyframes observing the point and associated index in keyframe
	std::map<KeyFrame*,size_t> mObservations;

	// Mean viewing direction
	cv::Mat mNormalVector;

	// Best descriptor to fast matching
	cv::Mat mDescriptor;

	// Reference KeyFrame
	KeyFrame* mpRefKF;

	// Tracking counters
	int mnVisible;
	int mnFound;

	// Bad flag (we do not currently erase MapPoint from memory)
	bool mbBad;
	ObjectPoint* mpReplaced;

	// Scale invariance distances
	float mfMinDistance;
	float mfMaxDistance;

	// Foreground / Background probability
	float mnFp;
	float mnBp;
	float mMinProb;

	std::mutex mMutexPos;
	std::mutex mMutexFeatures;
};

#endif // OBJECTPOINT_H
