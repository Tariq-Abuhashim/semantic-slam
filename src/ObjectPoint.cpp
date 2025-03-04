
/*
 * @file ObjectPoint.cpp
 * This is part of Semantic SLAM.
 * Functions to define and handle object 2D/3D points data.
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-07-00
 */

#include <ObjectPoint.hpp>

long unsigned int ObjectPoint::nNextId=0;
//mutex ObjectPoint::mGlobalMutex;

ObjectPoint::ObjectPoint(const cv::Mat &Pos, KeyFrame *pRefKF):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<ObjectPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0),
	mnFp(0.0), mnBp(0.0), mMinProb(0.75)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // ObjectPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    //unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void ObjectPoint::SetWorldPos(const cv::Mat &Pos)
{
    //unique_lock<mutex> lock2(mGlobalMutex);
    //unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat ObjectPoint::GetWorldPos()
{
    //unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

//cv::Mat ObjectPoint::GetNormal()
//{
    //unique_lock<mutex> lock(mMutexPos);
//    return mNormalVector.clone();
//}

KeyFrame* ObjectPoint::GetReferenceKeyFrame()
{
    //unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void ObjectPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    //unique_lock<mutex> lock(mMutexFeatures);
   	if(mObservations.count(pKF))
        return;
    mObservations[pKF]=idx;
	nObs++;
}

/*
void ObjectPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        //unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}
*/

map<KeyFrame*, size_t> ObjectPoint::GetObservations()
{
    //unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int ObjectPoint::Observations()
{
    //unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

/*
void ObjectPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
}
*/

ObjectPoint* ObjectPoint::GetReplaced()
{
    //unique_lock<mutex> lock(mMutexFeatures);
    return mpReplaced;
}

void ObjectPoint::Replace(ObjectPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        //unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
}

bool ObjectPoint::isBad()
{
    //unique_lock<mutex> lock(mMutexFeatures);
    //return mbBad;
	return (GetPointProbability()<mMinProb);
}

void ObjectPoint::IncreaseVisible(int n)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void ObjectPoint::IncreaseFound(int n)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float ObjectPoint::GetFoundRatio()
{
    //unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}


int ObjectPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool ObjectPoint::IsInKeyFrame(KeyFrame *pKF)
{
    //unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void ObjectPoint::UpdateForegroundProbability(float score)
{
	//unique_lock<mutex> lock(mMutexFeatures);
	mnFp+=score;
}

void ObjectPoint::UpdateBackgroundProbability(float score)
{
	//unique_lock<mutex> lock(mMutexFeatures);
	//mnBp+=(1-score);
	mnBp+=score;
}

float ObjectPoint::GetForegroundProbability()
{
	//unique_lock<mutex> lock(mMutexFeatures);
	return mnFp;
}

float ObjectPoint::GetBackgroundProbability()
{
	//unique_lock<mutex> lock(mMutexFeatures);
	return mnBp;
}

float ObjectPoint::GetPointProbability()
{
	//unique_lock<mutex> lock(mMutexFeatures);
	return mnFp/(mnFp+mnBp);
}

void ObjectPoint::SetProbabilityThreshold(float ProbThd)
{	
	//unique_lock<mutex> lock(mMutexFeatures);
	mMinProb = ProbThd;
}
