/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

/* For mkdir */
#include <sys/stat.h>
#include <sys/types.h>

#include <unordered_map>

namespace ORB_SLAM2
{

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
		unique_lock<mutex> lock(mMutexReset);
		if(mbReset)
		{
		    mpTracker->Reset();
		    mbReset = false;
		}
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();

    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || 
			mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

	/* I've commented these lines out because they freeze the terminal
 	* Tariq Abuhashim - 23 / 03 / 2018
 	*/
    //if(mpViewer)
    //    pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin(); // reference keyframe of the frame
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin(); // timestamp of the frame
    list<bool>::iterator lbL = mpTracker->mlbLost.begin(); // tracking flag of the frame, which is true when tracking is lost (mState==LOST)
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++) // (mlRelativeFramePoses) contains a Frame pose relative to its keyframe in (mlpReferences)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

/* Compute the matrix product R = AB */
void System::matrix_product(int Am, int An, int Bm, int Bn, 
                    const float *A, const float *B, float *R) {
    int r = Am;
    int c = Bn;
    int m = An;
    int i, j, k;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            R[i * c + j] = 0.0;
            for (k = 0; k < m; k++) {
                R[i * c + j] += A[i * An + k] * B[k * Bn + j];
            }
        }
    }
}

/* Scale a matrix by a scalar */
void System::matrix_scale(int m, int n, float *A, float s, float *R) {
    int i;
    int entries = m * n;
    
    for (i = 0; i < entries; i++) {
	R[i] = A[i] * s;
    }
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " 
			<< t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " " 
			<< q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) 
			<< " "  << twc.at<float>(0) << " " << Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  
			<< " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " << Rwc.at<float>(2,0) 
			<< " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}


/* Saving KeyFrame camera matrices for PMVS2
 * Tariq Abuhashim - 09 / 04 / 2018
 * FIXME: Review needed
 */
void System::WritePMVS(const char *output_path, vector<string> vstrimages, const string &strSettingsFile, bool flag_rectify)
{
    
	cout << endl << "Saving camera trajectory to " << output_path << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: WritePMVS cannot be used for monocular." << endl;
        return;
    }
    
    /* Make sure output_path exist */
    cout << "Creating folders ..." << endl;
    char command[256];
    sprintf(command, "mkdir -p %s/models/\n", output_path);
    int dir_err = system(command);
	if (-1 == dir_err)
	{
		printf("Error creating directory!n");
		exit(1);
	}
	sprintf(command, "mkdir -p %s/txt/\n", output_path);
	dir_err = system(command);
	sprintf(command, "mkdir -p %s/visualize/\n", output_path);
	dir_err = system(command);
	
    /* Sort KeyFrames */
    cout << "Preprocessing KeyFrames ..." << endl;
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
    // mapping old KeyFrame image Id to a new sequential Id
    std::map<int,int> KFId;
    int count = 0;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
    	//FIXME: Have to check if its a valid KeyFrame
    	if (vpKFs[i]->isBad()) 
    		continue;
    	KFId[vpKFs[i]->mnFrameId] = count;
    	count++;
    }

	/* Frames and poses */
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();
    
    // Generate undistortion map
    // Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }
    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["LEFT.P"] >> P_l;
    fsSettings["LEFT.R"] >> R_l;
    fsSettings["LEFT.D"] >> D_l;
    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    cv::Mat M1l,M2l;
    if (flag_rectify && !K_l.empty())
    {
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    }
    
    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.
	
    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    //list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin(); //reference keyframe of the frame
    //list<double>::iterator lT = mpTracker->mlFrameTimes.begin(); // timestamp of the frame
    //list<bool>::iterator lbL = mpTracker->mlbLost.begin(); // tracking flag of the frame, which is true when tracking is lost (mState==LOST)
    //count = 0;
    
    cout << "Generating camera matrices and undistorted KeyFrames in PMVS/txt and PMVS/visualize ..." << endl;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        
        KeyFrame* pKF = vpKFs[i];
          
        if (pKF->isBad())
        	continue;
        
        /* Write matrix projection matrix to a file */
   
        char buf1[256];
        sprintf(buf1, "%s/txt/%08d.txt", output_path, KFId[pKF->mnFrameId]); // used to be using "count"
        FILE *f = fopen(buf1, "w");
        assert(f);
        
        //cv::Mat Tcw = pKF->GetPose();
        //cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        //cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);
        
        cv::Mat Twc = pKF->GetPose();
        cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
        cv::Mat twc = Twc.rowRange(0,3).col(3);
        
		float K[9] =
			{ pKF->fx, 0.0, pKF->cx,
              0.0, pKF->fy, pKF->cy,
              0.0, 0.0, 1.0 };
        float Ptmp[12] = 
            { Rwc.at<float>(0,0), Rwc.at<float>(0,1), Rwc.at<float>(0,2), twc.at<float>(0),
              Rwc.at<float>(1,0), Rwc.at<float>(1,1), Rwc.at<float>(1,2), twc.at<float>(1),
              Rwc.at<float>(2,0), Rwc.at<float>(2,1), Rwc.at<float>(2,2), twc.at<float>(2) };
        float P[12];
        matrix_product(3, 3, 3, 4, K, Ptmp, P);
        //matrix_scale(3, 4, P, -1.0, P);  // FIXME
        fprintf(f, "CONTOUR\n");
        fprintf(f, "%0.6f %0.6f %0.6f %0.6f\n", P[0], P[1], P[2],  P[3]);
        fprintf(f, "%0.6f %0.6f %0.6f %0.6f\n", P[4], P[5], P[6],  P[7]);
        fprintf(f, "%0.6f %0.6f %0.6f %0.6f\n", P[8], P[9], P[10], P[11]);
        fclose(f);
        
        /* Write image to a file */
        cv::Mat imLeft;
        //std::cout << vstrimages[pKF->mnFrameId] << "\n";
        imLeft = cv::imread(vstrimages[pKF->mnFrameId]);
        if (flag_rectify && !K_l.empty())
        {
        	cv::remap(imLeft, imLeft, M1l, M2l, cv::INTER_LINEAR);
        }
        char buf2[256];
  		sprintf (buf2, "%s/visualize/%08d.ppm", output_path, KFId[vpKFs[i]->mnFrameId]);//pKF->mnId);
  		vector<int> compression_params;
    	compression_params.push_back(CV_IMWRITE_PXM_BINARY);
    	compression_params.push_back(1);
        cv::imwrite( buf2, imLeft, compression_params);
		
	}
	
	/* Write covisibility graph to a file */
    cout << "Generating vis.dat from covisibility graph ..." << endl;
    ofstream fg;
    char buf[256];
    sprintf(buf, "%s/vis.dat", output_path);
   	fg.open(buf);
    fg << fixed;
	for(size_t i=0; i<vpKFs.size(); i++)
    {
    	if (vpKFs[i]->isBad()) {
        	continue;
		}
		// Covisibility Graph
		const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
		if (!vCovKFs.empty())
		{
			fg << KFId[vpKFs[i]->mnFrameId] << " " <<  vCovKFs.size();
			
			for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
			{
 				//if((*vit)->mnId<vpKFs[i]->mnId) // Commented out because the Graph has to be Symmetric
				//	continue;
				fg <<  " " << KFId[(*vit)->mnFrameId];
			}
			
			fg <<  " " << endl;
		}
	}
	fg.close();
	
	/* Write PMVS option.txt file */
	cout << "Generating PMVS option.txt file ..." << endl;
    char option_buffer[2048];
    sprintf(option_buffer, "%s/option.txt", output_path);
    FILE *f_option = fopen(option_buffer, "w");
    fprintf(f_option, "level %d \n", 2);
    fprintf(f_option, "csize %d \n", 2);
    fprintf(f_option, "threshold %0.2f \n", 0.7);
    fprintf(f_option, "wsize %d \n", 7);
    fprintf(f_option, "minImageNum %d \n", 3);
    fprintf(f_option, "CPU %d \n", 4);
    fprintf(f_option, "useVisData %d \n", 1);
    fprintf(f_option, "sequence %d \n", -1);
    fprintf(f_option, "timages %d %d %d\n", -1, 0, count);
    fprintf(f_option, "oimages %d \n", 0);
    fclose(f_option);
    
    /* Done */
    cout << endl << "ORB_SLAM to PMVS formating all done !" << endl;
}

/* Saving KeyFrame camera matrices, images and 3D scans
 * Tariq Abuhashim - 29 / 10 / 2018
 * FIXME: Review needed
 */
void System::WriteRGBD( const char *output_path, vector<string> &vstrimages, vector<string> &vstrdepth, const string &strSettingsFile )
{

	cout << endl << "Saving RGBD data to " << output_path << " ..." << endl;
    if(mSensor!=System::RGBD)
    {
        cerr << "ERROR: WriteRGBD can only be used for Sensor=RGBD." << endl;
        return;
    }

	/* Make sure output_path exist */
    char command[256];
    sprintf(command, "mkdir -p %s/scans/\n", output_path);
	sprintf(command, "mkdir -p %s/poses/\n", output_path); system(command);
	sprintf(command, "mkdir -p %s/rgb/\n", output_path); system(command);
	sprintf(command, "mkdir -p %s/depth/\n", output_path); system(command);

	/* Sort KeyFrames */
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // mapping old KeyFrame image Id to a new sequential Id
    std::map<int,int> KFId;
    int count = 0;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
    	//FIXME: Have to check if its a valid KeyFrame
    	if (vpKFs[i]->isBad()) {
    		continue;
		}
    	KFId[vpKFs[i]->mnFrameId] = count;
    	count++;
    }

	cv::Mat Two = vpKFs[0]->GetPoseInverse();

	// For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
	cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    float DepthMapFactor;
    DepthMapFactor = fsSettings["DepthMapFactor"];
    if(fabs(DepthMapFactor)<1e-5)
		DepthMapFactor=1;
	else
		DepthMapFactor = 1.0f/DepthMapFactor;

	//cv::Mat DistCoef(4,1,CV_32F);
    //DistCoef.at<float>(0) = fsSettings["Camera.k1"];
    //DistCoef.at<float>(1) = fsSettings["Camera.k2"];
    //DistCoef.at<float>(2) = fsSettings["Camera.p1"];
    //DistCoef.at<float>(3) = fsSettings["Camera.p2"];
    //const float k3 = fsSettings["Camera.k3"];
    //if(k3!=0)
    //{
    //    DistCoef.resize(5);
    //    DistCoef.at<float>(4) = k3;
    //}

	// For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
	cv::Mat imRGB, imD;
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad()) {
        	continue;
		}

		cv::Mat Tcw = pKF->GetPose();
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

		// Read image and depthmap from file
		imRGB = cv::imread(vstrimages[pKF->mnFrameId],CV_LOAD_IMAGE_UNCHANGED);
		imD = cv::imread(vstrdepth[pKF->mnFrameId],CV_LOAD_IMAGE_UNCHANGED);

    	//cv::Mat mImGray = imRGB;
    	cv::Mat imDepth = imD;

    	if((fabs(DepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F) {
        	imDepth.convertTo(imDepth,CV_32F,DepthMapFactor);
		}

		//ofstream f;
		//char buf1[256];
        //sprintf(buf1, "%s/scans/%08d.txt", output_path, KFId[pKF->mnFrameId]); // used to be using "count"
    	//f.open(buf1);
    	//f << fixed;
		//for (int row=0; row < imDepth.rows; row++) {
		//	for (int col=0; col < imDepth.cols; col++) {
		//		const float z = imDepth.at<float>(row, col);
		//		if (z>0)
		//		{
		//			// Undistort points
		//			cv::Mat mat(1,2,CV_32F);
		//			mat.at<float>(0)=row;
		//			mat.at<float>(1)=col;
		//			mat=mat.reshape(2);
		//	   	 	cv::undistortPoints(mat,mat,pKF->mK,DistCoef,cv::Mat(),pKF->mK);
		//			mat=mat.reshape(1);
		//			// reproject points
		//			const float v = mat.at<float>(0);
		//			const float u = mat.at<float>(1);
		//			const float x = (u-pKF->cx)*z*pKF->invfx;
		//			const float y = (v-pKF->cy)*z*pKF->invfy;
		//			cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
		//			x3Dc = Rwc*x3Dc+twc;
		//			f<<setprecision(9)<<x3Dc.at<float>(0)<<" "<<x3Dc.at<float>(1)<<" "<<x3Dc.at<float>(2)<<endl;
		//		}
		//	}
		//}
		//f.close();

		/* Write image to a file */
        char buf2[256];
  		sprintf (buf2, "%s/rgb/%08d.png", output_path, KFId[vpKFs[i]->mnFrameId]);//pKF->mnId);
  		vector<int> compression_params;
    	compression_params.push_back(CV_IMWRITE_PXM_BINARY);
    	compression_params.push_back(1);
        cv::imwrite( buf2, imRGB, compression_params);

		sprintf (buf2, "%s/depth/%08d.png", output_path, KFId[vpKFs[i]->mnFrameId]);//pKF->mnId);
		cv::imwrite( buf2, imD, compression_params);

	}
	cout << endl << "RGB-D data has been saved!" << endl;
} // WriteRGBD

/* Saving Map Points to file
 * Tariq Abuhashim - 30 / 11 / 2018
 * FIXME: Review needed
 */
void System::SaveMapPoints(const string &filename)
{
	cout << endl << "Saving map points to " << filename << " ..." << endl;
    
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();

    if(vpMPs.empty())
        return;

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad())
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        f << setprecision(9) << pos.at<float>(0) << " " << pos.at<float>(1) << " " << pos.at<float>(2) << endl;
    }

	f.close();
}

/* Saving Data to bundler file format
 * Tariq Abuhashim - July, 2019
 */
void System::SaveMap(const string &filename)
{

	ofstream f;
    f.open(filename.c_str());
    f << fixed;

	vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
	sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
	unordered_map<size_t,size_t> NewId;

	const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();

	// <#KeyFrames> <#MapPoints>
	f << vpKFs.size() << " " << vpMPs.size() << endl;

	size_t count=0;
	for(size_t i=0; i<vpKFs.size(); i++) {  // KeyFrames are re-numbered as we go in the loop

		if(vpKFs[i]->isBad()) continue;
		cv::Mat Twc = vpKFs[i]->GetPose();
    	cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
    	cv::Mat twc = Twc.rowRange(0,3).col(3);
        
		// <KeyFrame#i>
		f << 0.0 << " " << 0.0 << " " << 0.0 << endl;
        f << setprecision(6) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1) << " " << Rwc.at<float>(0,2) << endl;
        f << setprecision(6) << Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1) << " " << Rwc.at<float>(1,2) << endl;
        f << setprecision(6) << Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1) << " " << Rwc.at<float>(2,2) << endl;
		f << setprecision(6) << twc.at<float>(0)   << " " << twc.at<float>(1)   << " " << twc.at<float>(2)   << endl;
		
		// remap KeyFrame->mnId
		NewId.insert(make_pair(vpKFs[i]->mnId,count)); // To insert KeyFrames in the same numbered order
		count++; // FIXME: However, check that count=0 is the KeyFrame [I,0]

	}

	for(size_t j=0; j<vpMPs.size(); j++) {

		if(vpMPs[j]->isBad()) continue;
        cv::Mat pos = vpMPs[j]->GetWorldPos();
        map<KeyFrame*, size_t> Observations = vpMPs[j]->GetObservations();
		// <MapPoint#j>
		// <Coordinates>
		f << setprecision(6) << pos.at<float>(0) << " " << pos.at<float>(1) << " " << pos.at<float>(2) << endl;
		// <Color>
		f << 0 << " " << 0 << " " << 0 << endl;
		// <#Views> <KeyFrame#i> <KeyPoint#j> <x> <y> ..... 
		//f << vpMPs[j]->Observations();
		f << Observations.size();
		for(auto itr = Observations.begin(); itr != Observations.end(); itr++) {
			KeyFrame* kf = itr->first;
			cv::KeyPoint kp = kf->mvKeys[itr->second];
			f << " " << NewId[kf->mnId] << " " << itr->second << " " << setprecision(6) << kp.pt.x << " " << kp.pt.y;
		}
		f << endl;
		
	}

	f.close();

}

/* Saving image KeyPoints to coords.txt file format
 * Tariq Abuhashim - July, 2019
 */
void System::SaveCoords(const string &filename)
{
	ofstream f;
    f.open(filename.c_str());
    f << fixed;

	vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
	sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

	size_t count=0;
	for(size_t i=0; i<vpKFs.size(); i++) { // KeyFrames are re-numbered as we go in the loop

		if(vpKFs[i]->isBad()) continue;
		float px=vpKFs[i]->cx, py=vpKFs[i]->cy, fx=vpKFs[i]->fx;
		vector<cv::KeyPoint> vKeys = vpKFs[i]->mvKeys;
		size_t nKeys = vKeys.size();
		f << "#index = " << count << ", name = rgb/xxxxx.jpg, keys = " << nKeys 
			<< ", px = " << px << ", py = " << py << ", focal = " << fx << endl;
		for(size_t j=0; j<nKeys; j++) {
			f << j << " " << vKeys[j].pt.x << " " << vKeys[j].pt.y << " 0 0 r g b" << endl;
		}
		count++;

	}

	f.close();
}

/* Saving image KeyPoints to coords.txt file format
 * Tariq Abuhashim - July, 2019
 */
void System::SaveAssociations(const string &filename,vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
	ofstream f;
    f.open(filename.c_str());
    f << fixed;

	vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
	sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

	size_t count=0;
	for(size_t i=0; i<vpKFs.size(); i++) { // KeyFrames are re-numbered as we go in the loop

		if(vpKFs[i]->isBad()) continue;
		int mnFrameId = vpKFs[i]->mnFrameId;
		f << vTimestamps[mnFrameId] << " " << vstrImageFilenamesRGB[mnFrameId] << " " 
			<< vTimestamps[mnFrameId] << " " << vstrImageFilenamesD[mnFrameId] << endl;

	}

	f.close();
}

/* Get Map
 * Tariq Abuhashim - July, 2019
 */
void System::GetMap(Map* Map) 
{
	Map = mpMap;
}

/* Saving Covisibility Graph for PMVS2
 * Tariq Abuhashim - 23 / 03 / 2018
 */
void System::SaveCovisibilityGraph(const string &filename)
{
	cout << endl << "Saving covisibility graph to " << filename << " ..." << endl;
    
    ofstream f;
    f.open(filename.c_str());
    f << fixed;
    
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
    
    std::map<int,int> KFId; // mapping old KeyFrame image Id to a new sequential Id.
    for(size_t i=0; i<vpKFs.size(); i++)
    {
    	KFId[vpKFs[i]->mnId]=i;
    }
    
    for(size_t i=0; i<vpKFs.size(); i++)
    {
		// Covisibility Graph
		//const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
		const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetVectorCovisibleKeyFrames();
		if(!vCovKFs.empty())
		{
			f << KFId[vpKFs[i]->mnId] << " " <<  vCovKFs.size();
			
			for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
			{
 				//if((*vit)->mnId<vpKFs[i]->mnId) // Commented out because the Graph has to be Symmetric
				//	continue;
				f <<  " " << KFId[(*vit)->mnId];
			}
			
			f <<  " " << endl;
		}
	}
	f.close();
}

/* Saving Camera Graph
 * Tariq Abuhashim - 24 / 09 / 2019
 */
void System::SaveCameraGraph(const string &filename)
{
	cout << endl << "Saving covisibility graph to " << filename << " ..." << endl;
    
    ofstream f;
    f.open(filename.c_str());
    f << fixed;
    
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
    
    std::map<int,int> KFId; // mapping old KeyFrame image Id to a new sequential Id.
    for(size_t i=0; i<vpKFs.size(); i++)
    {
    	KFId[vpKFs[i]->mnId]=i;
    }
    
    for(size_t i=0; i<vpKFs.size(); i++)
    {
		KeyFrame* pKF = vpKFs[i];
		// Covisibility Graph
		//const vector<KeyFrame*> vCovKFs = pKF->GetCovisiblesByWeight(100);
		const set<KeyFrame*> sKFs = pKF->GetConnectedKeyFrames();
		//const vector<KeyFrame*> vCovKFs = pKF->GetVectorCovisibleKeyFrames();

		if(!sKFs.empty())
		{
			int count = 0;
			for(set<KeyFrame*>::const_iterator sit=sKFs.begin(), send=sKFs.end(); sit!=send; sit++)
				if((*sit)->mnId < pKF->mnId) // Opposite to ORB-SLAM graph (object search looks back in time)
					count++;

			f << KFId[pKF->mnId] << " " <<  count;
			
			for(set<KeyFrame*>::const_iterator sit=sKFs.begin(), send=sKFs.end(); sit!=send; sit++)
			{
 				if((*sit)->mnId < pKF->mnId) // Opposite to ORB-SLAM graph (object search looks back in time)
					f <<  " " << KFId[(*sit)->mnId] << " " << pKF->GetWeight(*sit);
			}
			
			f << endl;
		}
	}
	f.close();
}


void System::SaveFrameId(const string &filename)
{
	ofstream f;
    f.open(filename.c_str());
    f << fixed;

	vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
	sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

	for(size_t i=0; i<vpKFs.size(); i++) { // KeyFrames are re-numbered as we go in the loop

		if(vpKFs[i]->isBad()) continue;
		f << vpKFs[i]->mnFrameId << endl;
	}

	f.close();
}

} //namespace ORB_SLAM
