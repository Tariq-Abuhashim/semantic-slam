
#ifndef OBJECTDRAWER_H
#define OBJECTDRAWER_H

#include "Thirdparty/ORB_SLAM2/include/KeyFrame.h"

#include <pangolin/pangolin.h>

#include "Inventory.hpp"
#include "Object.hpp"
#include "ObjectPoint.hpp"

#include <mutex>

class ObjectDrawer
{
public:
    ObjectDrawer(Inventory* pInventory, const string &strSettingPath);

    Inventory* mpInventory;

    void DrawMapPoints();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SetCurrentCameraPose(const cv::Mat &Tcw);
    void SetReferenceKeyFrame(KeyFrame *pKF);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

private:

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    cv::Mat mCameraPose;

    std::mutex mMutexCamera;
};

#endif // OBJECTDRAWER_H
