
/*
 * @file InstanceViewer.cpp
 * This is part of Semantic SLAM.
 * Functions to visualise object instances.
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-07-00
 */

#include "InstanceViewer.hpp"
#include <pangolin/pangolin.h>

#include <mutex>

InstanceViewer::InstanceViewer(ObjectDrawer *pObjectDrawer, const string &strSettingPath):
    mpObjectDrawer(pObjectDrawer), mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    mImageWidth = fSettings["Camera.width"];
    mImageHeight = fSettings["Camera.height"];
    if(mImageWidth<1 || mImageHeight<1)
    {
        mImageWidth = 640;
        mImageHeight = 480;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];
}

void InstanceViewer::Run()
{
    //mbFinished = false;
    //mbStopped = false;

	// Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Safe and efficient binding of named variables.
  	// Specialisations mean no conversions take place for exact types
  	// and conversions between scalar types are cheap.
    pangolin::CreatePanel("gui").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("gui.Follow Camera",true,true);
    pangolin::Var<bool> menuShowPoints("gui.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("gui.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("gui.Show Graph",true,true);
    //pangolin::Var<bool> menuLocalizationMode("gui.Localization Mode",false,true);
    //pangolin::Var<bool> menuReset("gui.Reset",false,false);
	pangolin::Var<bool> menuSave("gui.Save",false,false);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,-1.0,0.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    //cv::namedWindow("Current Frame");

    bool bFollow = true;
    bool bLocalizationMode = false;

    while(!pangolin::ShouldQuit())
    {

		// Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpObjectDrawer->GetCurrentOpenGLCameraMatrix(Twc);

        if(menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc);
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

		// Examples from https://github.com/stevenlovegrove/Pangolin/blob/master/examples/SimpleDisplay/main.cpp
		//if( pangolin::Pushed(save_window) )
        //pangolin::SaveWindowOnRender("window");
    	//if( pangolin::Pushed(save_cube) )
        //d_cam.SaveOnRender("cube");
    	//if( pangolin::Pushed(record_cube) )
        //pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi");

		if(menuSave)
        {
			std::cout << "Render saved " << std::endl;
			//pangolin::SaveWindowOnRender("window");
			d_cam.SaveOnRender("Map");
			menuSave = false;
        }
		
		// Activate efficiently by object
		d_cam.Activate(s_cam);

		// Render some stuff
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        mpObjectDrawer->DrawCurrentCamera(Twc);
        if(menuShowKeyFrames || menuShowGraph)
            mpObjectDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph);
        if(menuShowPoints)
            mpObjectDrawer->DrawMapPoints();

		// Swap frames and Process Events
        pangolin::FinishFrame();

        //cv::Mat im = mpFrameDrawer->DrawFrame();
        //cv::imshow("Current Frame",im);
        //cv::waitKey(mT);

        //if(Stop())
        //{
        //    while(isStopped())
        //    {
        //        usleep(3000);
        //    }
        //}

        //if(CheckFinish())
        //    break;

    }

	//d_cam.SaveOnRender("image");

	//cv::Mat img, imgBuffer = cv::Mat(d_cam.h, d_cam.w, CV_8UC4, buffer.ptr);
    //cv::cvtColor(imgBuffer, img,  cv::COLOR_RGBA2BGR);
    //cv::flip(imagen, img, 0);
    //cv::imshow("some window", img);

	cv::waitKey(0);

    // SetFinish();
}

void InstanceViewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool InstanceViewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void InstanceViewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool InstanceViewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void InstanceViewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool InstanceViewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool InstanceViewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void InstanceViewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}
