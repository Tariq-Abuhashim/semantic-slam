
/*
 * @file Engine.cpp
 * This is part of Semantic SLAM.
 * Semantic SLAM engine.
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-07-16
 */

#include "Engine.hpp"
#include <map>

Engine::Engine(unordered_map<int,string> categories, string strSettingsFile) : mmCategories(categories)
{

	cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);

    //float fps = fSettings["Camera.fps"];
    //if(fps<1) fps=30;
    //mT = 1e3/fps;

    mWidth = fSettings["Camera.width"];
    mHeight = fSettings["Camera.height"];

	float fx = fSettings["Camera.fx"];
	float fy = fSettings["Camera.fy"];
	float cx = fSettings["Camera.cx"];
	float cy = fSettings["Camera.cy"];
	K = (cv::Mat_<float>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

	float k1 = fSettings["Camera.k1"];
	float k2 = fSettings["Camera.k2"];
	float p1 = fSettings["Camera.p1"];
	float p2 = fSettings["Camera.p2"];
	DistCoef = (cv::Mat_<float>(4,1) << k1, k2, p1, p2);

	//Tracking parameters
	mSensor = fSettings["Engine.Sensor"]; // 1-Lidar, 2-RGBD, 3-Stereo
	mnDist = fSettings["Engine.mnDist"]; // 2
	mnMinDepth = fSettings["Engine.mnMinDepth"];   // 0.2
	mnMaxDepth = fSettings["Engine.mnMaxDepth"];   // RGBD 3, Lidar 30	
	mMinArea = fSettings["Engine.mMinArea"];   // RGBD 3, Lidar 30
	mMaxArea = fSettings["Engine.mMaxArea"];   // 20*20 and 500*500
	mOverlap = fSettings["Engine.mOverlap"]; // 0.75
	mProbThd = fSettings["Engine.mProbThd"]; // 0.75
	mRes = fSettings["Engine.mRes"]; // Lidar 1 pixel, RGBD 4 pixels
	mMinPointCount = fSettings["Engine.mMinPointCount"]; // 15

	//Create a new Mask-RCNN (python on GPU or CPU) object
	mpMaskRCNN = new MaskRCNN();
	//mptMaskRCNN = new thread(&MaskRCNN::Run, mpMaskRCNN);

	//Create a new DoN segmentation object (previously, .2, 4.,.25,.2)
	double scale1 = fSettings["DoN.scale1"];
	double scale2 = fSettings["DoN.scale2"];
	double threshold = fSettings["DoN.threshold"];
	double segradius = fSettings["DoN.segradius"];
	mpDoN = new DoN(scale1, scale2, threshold, segradius, K, mWidth, mHeight, mSensor);

	//Create a new inventory
	mpInventory = new Inventory();

	//Create a new drawer window
	mpObjectDrawer = new ObjectDrawer(mpInventory, strSettingsFile);

	//Initialize the Viewer thread and launch
	mpViewer = new InstanceViewer(mpObjectDrawer, strSettingsFile);
	mptViewer = new thread(&InstanceViewer::Run, mpViewer);

}

Engine::~Engine()
{
	delete(mpMaskRCNN);
	delete(mpDoN);
	delete(mpInventory);
	delete(mpViewer);
}

void Engine::tick()
{
	mtimer = tick<timer>();
	timeIsTicked = true;
}

double Engine::tock()
{
	if(timeIsTicked)
	{
		timeIsTicked = false;
		return tock(mtimer);
	}
	else
	{
		std::cerr << " Error: You should tick() before you tock()." << std::endl;
	}
	return 0.0;
}

/*
*	Run
* 
*/
void Engine::Run(cv::Mat imRGB, cv::Mat imD, KeyFrame* KF)
{

	mpCurrentKF = KF;

	std::cout << "  *RGB image size : " << imRGB.cols << "x" << imRGB.rows << std::endl;
	std::cout << "  *D   image size : " << imD.cols << "x" << imD.rows << std::endl;

	// Geometric segmentation
	tick();
	std::map<int, std::vector<cv::Point> > geometry;
	if(true){
		geometry = mpDoN->extract(imRGB, imD);
		std::cout <<"  *Geometric segmentation "<< tock() <<"s, "<< geometry.size() <<" clusters."<< std::endl;
		mpDoN->show2d(imRGB);
	}
	//mpDoN->show3d();


	// Texture detection
	tick();
	mpMaskRCNN->Run(imRGB);
	std::vector<cv::Rect> boxes = mpMaskRCNN->GetBoxes();
	std::vector<cv::Mat> masks = mpMaskRCNN->GetMasks();
	std::vector<int> labels = mpMaskRCNN->GetLabels();
	std::vector<double> scores = mpMaskRCNN->GetScores();
	std::cout << "  *Texture detection " << tock() << "s, " << masks.size() << " masks." << std::endl;

	//for(int col=0; col<imDepth.cols; col++) {
	//	for(int row=0; row<imDepth.rows; row++) {
	//		float range = imDepth.at<float>(row,col);
	//		if (range>0.1f) {
	//			cv::Vec3b color = cv::Vec3b(range,range,range);
	//			cv::circle(imRGB, cv::Point(col,row), 1, 5*color, -1, 2); // depth scaled for visualisation
	//		}
	//	}
	//}

	mpMaskRCNN->show2d(imRGB, mmCategories);

	if(masks.size()==0 || geometry.size()==0) {
		std::cout << "  ****Warning: In Engine::Run(), masks.size() is 0 or geometry.size() is 0." << std::endl;
		return;
	}

	mpInventory->AddKeyFrame(mpCurrentKF);
	mpObjectDrawer->SetCurrentCameraPose(mpCurrentKF->GetPose());

	// Vector for tracking time statistics
    std::vector<double> step1;
    step1.resize( masks.size() );
	std::vector<double> step2;
    step2.resize( masks.size() );
	std::vector<double> step3;
    step3.resize( masks.size() );
	std::vector<double> step4;
    step4.resize( masks.size() );
	std::vector<double> step5;
    step5.resize( masks.size() );
	std::vector<double> step6;
    step6.resize( masks.size() );

	if(true)
	{

	#pragma omp parallel
	#pragma omp for
	for(size_t i=0; i<masks.size(); i++) {

		// 1. Find contours
		tick();
		Contour texture;
		std::vector<cv::Vec4i> hierarchy;
		findContours( masks[i], texture, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

		float area = contourArea(texture[0]); // outer most contour only (we expect only one object in the mask)
		if (area<mMinArea || area>mMaxArea) continue; // small or large objects (ignore)
		step1[i]=tock();

		// 2. Check previous objects using points
		tick();
		Object *instance = TrackObjectPoints( texture, mmCategories[labels[i]], scores[i], imD );
		//Object *O2 = TrackObjectContours(maskContour, mmCategories[labels[i]], scores[i], imD, KF);
		step2[i]=tock();

		// 3. Check for a new object
		tick();
		cv::Mat masked( imD.size(), imD.type() ); // used to update the TSDF
		multiply( imD, masks[i]/255, masked, 1, imD.type() );
		bool ObjFound = true;
		if(instance==nullptr) {
			ObjFound = false;
			instance = new Object( mpCurrentKF, boxes[i], masked, K, mSensor );
			instance->SetLabel( mmCategories[labels[i]] );
			instance->UpdateScore( scores[i] );
			instance->SetObjectParams(mProbThd,mMinPointCount,mHeight,mWidth,mnMinDepth,mnMaxDepth,mnDist,mRes);
		}
		instance->AddKeyFrame( mpCurrentKF );
		instance->AddObservation( mpCurrentKF, instance->GetNumberOfContours() );
		instance->AddContour( texture );
		instance->AddBoundingBox( boxes[i] );
		step3[i]=tock();

		// 4. Add new data to an instance in the inventory
		tick();
		//O->AddNewObjectPoints( texture, imDepth, mpCurrentKF, scores[i] ); // MaskRCNN
		//O->AddNewObjectPoints( texture, geometry, imDepth, mpCurrentKF, scores[i] ); // MaskRCNN + DoN
		if(false) {
			instance->AddSegment( texture, imD, mpCurrentKF, scores[i] ); // MaskRCNN
		} else {
			std::vector<cv::Point> total_segment;
			total_segment = fuse_segments ( geometry, texture );
			instance->AddSegment( total_segment, imD, mpCurrentKF, scores[i] ); // MaskRCNN + DoN
		}
		step4[i]=tock();

		// TODO: Update TSDF or Voxel hashing, or PCL voxels ?
		//Mat Twc = KF->GetPoseInverse();
		//Mat masked(imDepth.size(), imDepth.type());
		//multiply(imDepth, masks[i]/255, masked, 1, imDepth.type());
		//double min, max;
		//cv::minMaxIdx(masked, &min, &max);
		//cout << "min/max : " << min << "/" << max << endl;
		//cv::Mat adjMap;
		//cv::convertScaleAbs(masked, adjMap, 255 / max);
	  	//imshow( "mask", adjMap );
		//O->Integrate(masked, Twc); // using Andy's cpp code
		//O->Integrate(imRGB, masked, Twc); // using Andy's python code (this takes color too)
		//cout << "Done !" << endl;

		// 5. Update instance in the inventory (Add or Delete ?)
		tick();
		std::vector<ObjectPoint*> vObjectPoints = instance->GetAllObjectPoints();
		if( !ObjFound && instance->HasEnoughObjectPoints() ) {
			mpInventory->AddObject( instance ); // if its a new Object insert to inventory
			//cout << " [Mask " << i << "] contains a new object: " << instance->GetLabel();
			//cout << " with " << vObjectPoints.size() << " ObjectPoints. "<< endl;
		}
		else if ( instance->HasEnoughObjectPoints() ) {
			//cout << " [Mask " << i << "] contains an old object: " << instance->GetLabel();
			//cout << " with " << vObjectPoints.size() << " ObjectPoints." << endl;
		}
		else {
			delete( instance );
			continue;
		}
		step5[i]=tock();

		// 6. Draw
		tick();
		if(false) {
			if(!ObjFound)
				display( instance, imRGB, texture, boxes[i], hierarchy, cv::Scalar( 0, 0, 255 ) );
			else
				display( instance, imRGB, texture, boxes[i], hierarchy, cv::Scalar( 0, 255, 0 ) );	
		}
		step6[i]=tock();

		//waitKey(0);

	} // masks loop

	float totaltime = 0;
    for(int ni=0; ni<step1.size(); ni++) {totaltime+=step1[ni];}
	std::cout << "  *Find contours average time :      " << totaltime/step1.size() << std::endl;

	totaltime = 0;
    for(int ni=0; ni<step2.size(); ni++) {totaltime+=step2[ni];}
	std::cout << "  *Track object average time :       " << totaltime/step2.size() << std::endl;

	totaltime = 0;
    for(int ni=0; ni<step3.size(); ni++) {totaltime+=step3[ni];}
	std::cout << "  *Create new object average time :  " << totaltime/step3.size() << std::endl;

	totaltime = 0;
    for(int ni=0; ni<step4.size(); ni++) {totaltime+=step4[ni];}
	std::cout << "  *Add object points average time :  " << totaltime/step4.size() << std::endl;
	
	totaltime = 0;
    for(int ni=0; ni<step5.size(); ni++) {totaltime+=step5[ni];}
	std::cout << "  *Insert object average time :      " << totaltime/step5.size() << std::endl;

	totaltime = 0;
    for(int ni=0; ni<step6.size(); ni++) {totaltime+=step6[ni];}
	std::cout << "  *Show detection average time :     " << totaltime/step6.size() << std::endl;
	std::cout << std::endl;

	} //if(false)

}

/*
*	Fuse Geometric Segments using the mask
* 
*/
std::vector<cv::Point> Engine::fuse_segments (std::map<int, std::vector<cv::Point> >& geometry, Contour texture)
{

	std::vector<cv::Point> total_segment;

	for(std::map<int, std::vector<cv::Point> >::const_iterator it=geometry.begin();it!=geometry.end();++it) 
	{
		std::vector<cv::Point> cluster = it->second;
		std::vector<cv::Point> temp;
		//float count = 0;

		for(int c=0; c<cluster.size(); c++)
		{
			float dist = -100;
			dist = pointPolygonTest(texture[0], cluster[c], true);	
			if(dist>mnDist) 
			{
				temp.push_back(cluster[c]);
				//count++;
			}
		}
		//if(count>0) {overlap[i]++;} // count this cvpoint in
		float overlap = (float)temp.size()/(float)cluster.size();
		//std::cout << overlap << std::endl;
		
		if(overlap>mOverlap)
		{
			//std::cout <<  it->first << " " << cluster.size() << " " << count << " " << overlap << std::endl;
			total_segment.reserve(total_segment.size() + distance(temp.begin(),temp.end()));
			total_segment.insert(total_segment.end(), temp.begin(), temp.end());
			//geometry.erase(it);
		}

		temp.clear();
		temp.resize(0);
	}

	return total_segment;
}

/*
*	IsInCurrentKF
* 
*/
bool Engine::IsInCurrentKF(float range, cv::Point cvpoint, int margin)
{
	return (cvpoint.x>margin && cvpoint.x<mWidth-margin && 
			cvpoint.y>margin && cvpoint.y<mHeight-margin &&
			range>0);
}


/*
*	TrackObjectPoints
* 
*/
Object* Engine::TrackObjectPoints(Contour maskContour, const string label, double score, cv::Mat imDepth) 
{

	const cv::Mat Twc = mpCurrentKF->GetPoseInverse();
	cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
	cv::Mat twc = Twc.rowRange(0,3).col(3);
	cv::Mat Rcw = Rwc.t();
	
	std::vector<Object*> Objects = mpInventory->GetAllObjects();

	for(auto sit=Objects.begin(); sit!=Objects.end(); sit++)
	{

		// Object candidate
		Object* object = *sit;
		int count = 0; // count number of inliers

		// point-based
		std::vector<ObjectPoint*> vObjectPoints=object->GetAllObjectPoints();//FIXME: to be replaced with TSDF points
		for(size_t MPi=0; MPi<vObjectPoints.size(); MPi++)
		{

			ObjectPoint* MP = vObjectPoints[MPi];
			//if(!MP->mbTrackInView) continue;
			if(MP->isBad()) continue; // only use reliably labeled points for tracking (0.75 lidar, 0.5 RGBD)

			cv::Mat x3Dc = MP->GetWorldPos();
			x3Dc = Rcw*(x3Dc-twc);
			float x2 = x3Dc.at<float>(0)*x3Dc.at<float>(0);
			float y2 = x3Dc.at<float>(1)*x3Dc.at<float>(1);
			float z2 = x3Dc.at<float>(2)*x3Dc.at<float>(2);
			float range = std::sqrt(x2+y2+z2);
			
			cv::Point cvpoint = ProjectIntoCurrentKF(MP);

			if (!IsInCurrentKF(x3Dc.at<float>(2),cvpoint,5)) continue;

			float p_hat = 1.0f/range;
			if (p_hat>1/mnMinDepth || p_hat<1/mnMaxDepth) continue; // ranges between mnMinDepth and mnMaxDepth 

			float p_m = 1.0f/imDepth.at<float>(cvpoint.y, cvpoint.x);
			float dp = (p_hat-p_m)/abs(p_hat-p_m); // FIXME: is this a good measure ?
				
			if (p_m<1/mnMaxDepth && p_m>1/mnMinDepth && dp>0.25)  continue; // FIXME: Occlusion assumption

			//for(size_t Ci = 0; Ci< maskContour.size(); Ci++)
			//{

			if(maskContour.size()>1) std::cout << "More than one contour" << std::endl;

			float dist = pointPolygonTest(maskContour[0], cvpoint, true);
			if (dist>mnDist) // && (label.compare(O->GetLabel())==0 || O->GetScore()>score))
			{
				count++;  // count number of MP in O that are in KF and in agreement with maskContour
				//break; // MP overlapping with the contour
			}
			//}

		} // points loop

		// contours-based
		//cout << "Mask-RCNN->" << label << " (" << score << ") " << endl;
		//cout << "O->" << O->GetLabel() << " (" << O->GetScore() << ") ";
		//cout << " has " << O->GetNumberOfContours() << " contours" << endl;
		size_t c = object->GetNumberOfContours();
		double minCnt=100.0;
		for(size_t i=0; i<c; i++)
		{
			//cout << "  ->Contour " << i << " ";
			Contour cnt = object->GetContour(i);
			for(size_t Ci=0; Ci<maskContour.size(); Ci++)
			{
				double matchR = matchShapes(cnt[0], maskContour[Ci], CV_CONTOURS_MATCH_I1, 0);
				if (matchR<minCnt) minCnt=matchR;
				//cout << matchR << " ";
			}
			//cout << endl;
		}
		//cout << endl;

		// decision on if maskContour belongs to Object (FIXME: modify rules according to literature)
		bool c1 = count > mMinPointCount; // Minimum 10 inlier points (FIXME: other places include Object::Object)
		bool c2 = minCnt < 1.0; // Contours similarity
		bool c3 = label.compare(object->GetLabel())==0; // The same label
		bool c4 = object->GetScore()>1.1f*score; // maybe different labels? but the candidate score should be higher

		//if(c1 && c2 && (c3 || c4))
		if(c1 && (c3 || c4))
		{
			//O->mnTracks = count;

			// loop points to update their probabilities
			std::vector<ObjectPoint*> vObjectPoints = object->GetAllObjectPoints();
			for(size_t MPi=0; MPi<vObjectPoints.size(); MPi++)
			{
				ObjectPoint* MP = vObjectPoints[MPi];
				//bool found = false;
				
				cv::Mat x3Dc = MP->GetWorldPos();
				x3Dc = Rcw*(x3Dc-twc);
				float x2 = x3Dc.at<float>(0)*x3Dc.at<float>(0);
				float y2 = x3Dc.at<float>(1)*x3Dc.at<float>(1);
				float z2 = x3Dc.at<float>(2)*x3Dc.at<float>(2);
				float range = std::sqrt(x2+y2+z2);

				cv::Point cvpoint = ProjectIntoCurrentKF(MP);

				if (!IsInCurrentKF(x3Dc.at<float>(2),cvpoint,5)) continue;

				float p_hat = 1.0f/range;
				if (p_hat>1.0f/mnMinDepth || p_hat<1.0f/mnMaxDepth) continue; 

				float p_m = 1/imDepth.at<float>(cvpoint.y, cvpoint.x);
				float dp = (p_hat-p_m)/abs(p_hat-p_m); // FIXME: is this a good measure ?
				
				if (p_m<1.0f/mnMaxDepth && p_m>1.0f/mnMinDepth && dp>0.25)  continue; // FIXME: Occlusion assumption

				float dist = pointPolygonTest(maskContour[0], cvpoint, true);
				if (dist>mnDist) // && (label.compare(O->GetLabel())==0 || O->GetScore()>score))
				{
					MP->UpdateForegroundProbability(score);
				} 
				else
				{
					MP->UpdateBackgroundProbability(score);
				}

			} // points loop

			//cout << "Match found : " << c1 << "-(" << count << "), ";
			//cout << c2 << "-(" << minCnt << "), " << c3 << ", " << c4 << endl;
			return object;
	
		} // Matching condition

	} // objects loop
	
	//cout << "[Point tracker] Match not found : " << count << endl;
	return nullptr;

}


/*
*	TrackObjectContours
* 
*/
Object* Engine::TrackObjectContours(Contour maskContour, const string label, double score, cv::Mat imDepth) 
{

	int count = 0;
	//bool ObjFound = false;

	std::vector<Object*> Objects = mpInventory->GetAllObjects();

	for(auto sit=Objects.begin(); sit!=Objects.end(); sit++)
	{

		Object* O = *sit;

		// Based on matching contour shapes - wont work well from two very different field of views 
		cout << "Mask-RCNN->" << label << " (" << score << ") " << endl;
		cout << "O->" << O->GetLabel() << " (" << O->GetScore() << ") ";
		cout << " has " << O->GetNumberOfContours() << " contours" << endl;
		size_t c = O->GetNumberOfContours();

		std::vector<KeyFrame*> vKeyFrames = O->GetAllKeyFrames();
		assert(vKeyFrames.size()==c);

		cv::Moments M1 = moments(maskContour[0],true); // KF1
		cv::Point centroid1(M1.m10/M1.m00, M1.m01/M1.m00); // KF1
		cv::Mat cnt1points = cv::Mat(maskContour[0]); // cv::Mat of cv::Point
		cv::Point top(0,0), bottom(0,0); // point defining the epipolar band
		for(int j=0; j<cnt1points.rows; j++)
		{
			cv::Point cp1 = cnt1points.at<cv::Point>(j,1); // contour point
			if (cp1.y>top.y) top=cp1;
			if (cp1.y<bottom.y) bottom=cp1;
		}


		double minCnt=100.0;
		for(size_t i=0; i<c; i++)
		{
			KeyFrame* KF = vKeyFrames[i];
			if(mpCurrentKF->mnId==KF->mnId) continue;

			Contour cnt = O->GetContour(KF); // this is overloaded usind KF or idx, but better to link it to KF
			cv::Mat F12 = ComputeFundamental(mpCurrentKF, KF); // mpCurrentKF to KF fundamental matrix

			// using Hu Moments
			double matchR = matchShapes(maskContour[0], cnt[0], CV_CONTOURS_MATCH_I1, 0);
			if (matchR<minCnt) minCnt=matchR;

			// using centroids // FIXME: should this be the TSDF center instead ?
			cv::Moments M2 = moments(cnt[0],true);
			cv::Point centroid2(M2.m10/M2.m00, M2.m01/M2.m00);
			// ax + by + c = 0, slope = -a/b, intersect = -c/b
			float a = centroid1.x*F12.at<float>(0,0)+centroid1.y*F12.at<float>(1,0)+F12.at<float>(2,0); 
			float b = centroid1.x*F12.at<float>(0,1)+centroid1.y*F12.at<float>(1,1)+F12.at<float>(2,1); 
			float c = centroid1.x*F12.at<float>(0,2)+centroid1.y*F12.at<float>(1,2)+F12.at<float>(2,2); 
			// shortest distance between centroid2 and epipolar line of centroid1 in KF2
			float d = shortest_distance(centroid2.x, centroid2.y, a, b, c); 

			// using boundary points
			//cv::Mat cnt2points = cv::Mat(cnt[0]); // cv::Mat of cv::Point
			int count = 0;
			//for(int j=0; j<cnt1points.rows; j++)
			//{
			//	Point cp1 = cnt1points.at<Point>(j,1); // contour point
			//	float a = cp1.x*F12.at<float>(0,0)+cp1.y*F12.at<float>(1,0)+F12.at<float>(2,0); // ax + by + c = 0
			//	float b = cp1.x*F12.at<float>(0,1)+cp1.y*F12.at<float>(1,1)+F12.at<float>(2,1); // slope = -a/b
			//	float c = cp1.x*F12.at<float>(0,2)+cp1.y*F12.at<float>(1,2)+F12.at<float>(2,2); // intersect = -c/b
			//	int umin = max(centroid2.x-100,3); // FIXME: similar to bounding box
			//	int umax = min(centroid2.x+100,imDepth.cols-3); // FIXME: similar to bounding box
			//	for(int u=umin; u<umax; u++) // FIXME: SLOWWWWWWW
			//	{
			//		int v = (-a/b)*u + (-c/b);
			//		if(v>0 && v<imDepth.rows)
			//		{
			//			float d_ = pointPolygonTest(cnt[0], Point(u,v), true);
			//			if(abs(d_)<.5) count++;
			//		}
			//	}
			//}

			cout << "  ->Contour " << i << " : " << " Hu Moments " << matchR;
			cout << ", EG centroid " << d << ", EG tangent " << count << "/" << cnt1points.rows << endl;
		}
		cout << endl;

		// decision on if maskContour belongs to Object
		//bool c1 = count>10;
		//bool c2 = minCnt<1.0;
		//bool c3 = label.compare(O->GetLabel())==0;
		//bool c4 = O->GetScore()>1.1*score;
		//if(c1 && c2 && (c3 || c4))
		//{
		//	O->mnTracks = count;
		//	cout << "Match found : " << c1 << "-(" << count << "), ";
		//  cout << c2 << "-(" << minCnt << "), " << c3 << ", " << c4 << endl;
		//	return O;
			//ObjFound = true;
			//break;
		//}

	}
	
	cout << "[Contour tracker] Match not found : " << count << endl;
	return nullptr;

}


/*
*	ProjectIntoCurrentKF
* 
*/
cv::Point Engine::ProjectIntoCurrentKF(ObjectPoint* MP)
{
	//unique_lock<mutex> lock(mMutexPose);

	const float fx= K.at<float>(0,0);
	const float fy= K.at<float>(1,1);
	const float cx= K.at<float>(0,2);
	const float cy= K.at<float>(1,2);

	const cv::Mat x3Dc = MP->GetWorldPos();
	const cv::Mat Twc = mpCurrentKF->GetPoseInverse();

	cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
    cv::Mat twc = Twc.rowRange(0,3).col(3);
    cv::Mat Rcw = Rwc.t();

	cv::Mat x2Dc = Rcw*(x3Dc-twc);
	const float x = x2Dc.at<float>(0)/x2Dc.at<float>(2);
	const float y = x2Dc.at<float>(1)/x2Dc.at<float>(2);
	const float u = fx*x + cx;
	const float v = fy*y + cy;

    return cv::Point(u,v);
}


/*
* 	Computes the fundamental matrix between two keyframes
* 	Checked against older version - Identical
*/
cv::Mat Engine::ComputeFundamental( KeyFrame* KF1,  KeyFrame* KF2) 
{
    cv::Mat R1w = KF1->GetRotation();
    cv::Mat t1w = KF1->GetTranslation();
    cv::Mat R2w = KF2->GetRotation();
    cv::Mat t2w = KF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w + t1w;

    cv::Mat t12x = GetSkewSymmetricMatrix(t12);

    return K.t().inv()*t12x*R12*K.inv();
}


/*
* 	Computes skew symetric matrix
* 	Checked against older version - Identical
*/
cv::Mat Engine::GetSkewSymmetricMatrix(const cv::Mat& v) 
{
    return (cv::Mat_<float>(3,3) <<            0, -v.at<float>(2), v.at<float>(1),
                              v.at<float>(2),               0,-v.at<float>(0),
                             -v.at<float>(1),  v.at<float>(0),             0);
}


/*
* 	Function to find distance
* 	Distance = (| a*x1 + b*y1 + c |) / (sqrt( a*a + b*b))
*/
float Engine::shortest_distance(float x, float y, float a, float b, float c) 
{ 
	return fabs((a * x + b * y + c)) /  
             (sqrt(a * a + b * b)); 
} 


/*
*	Display
* 
*/
void Engine::display(Object* object, cv::Mat imRGB, Contour maskContour, cv::Rect bbox, 
						std::vector<cv::Vec4i> hierarchy, cv::Scalar color)
{
	
	std::vector<ObjectPoint*> vObjectPoints = object->GetAllObjectPoints();
	if (vObjectPoints.size()==0) return;

	const cv::Mat Twc = mpCurrentKF->GetPoseInverse();
	cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
	cv::Mat twc = Twc.rowRange(0,3).col(3);
	cv::Mat Rcw = Rwc.t();

	std::cout << "Image points : " << vObjectPoints.size() << std::endl;

	//for( size_t Ci = 0; Ci< maskContour.size(); Ci++ )
	//{

	
		cv::drawContours( imRGB, maskContour, -1, color, 1, 4, hierarchy, 0, cv::Point() );
   //cv::rectangle(imRGB, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x+bbox.width, bbox.y+bbox.height), color, 2);
		std::string label = object->GetLabel()+" " +std::to_string(object->mnId);
		cv::putText(imRGB, label, cv::Point(bbox.x+2,bbox.y-2), cv::FONT_HERSHEY_DUPLEX, 0.5, color, 1);
		
		//for(size_t MPi=0; MPi<vObjectPoints.size(); MPi++)
		//{
		//	std::cout << MPi << std::endl;
		//	ObjectPoint* MP = vObjectPoints[MPi];
		//}

		for(size_t MPi=0; MPi<vObjectPoints.size(); MPi++)
		{
			ObjectPoint* MP = vObjectPoints[MPi];
			//if(!MP->mbTrackInView) continue;
			if(MP->isBad()) continue; // test point probability

			cv::Mat x3Dc = MP->GetWorldPos();
			x3Dc = Rcw*(x3Dc-twc);
			float x2 = x3Dc.at<float>(0)*x3Dc.at<float>(0);
			float y2 = x3Dc.at<float>(1)*x3Dc.at<float>(1);
			float z2 = x3Dc.at<float>(2)*x3Dc.at<float>(2);

			cv::Point cvpoint = ProjectIntoCurrentKF(MP);
			if (!IsInCurrentKF(x3Dc.at<float>(2),cvpoint,5)) continue;

			float range = std::sqrt(x2+y2+z2); // FIXME: Lidar
			float p_hat = 1/range;
			if (p_hat>1/mnMinDepth || p_hat<1/mnMaxDepth) continue; // ranges between mnMinDepth and mnMaxDepth 

			//float p_m = 1/imDepth.at<float>(cvpoint.y, cvpoint.x);
			//float dp = (p_hat-p_m)/abs(p_hat-p_m); // FIXME: is this a good measure ?	
			//if (p_m<1/mnMaxDepth && p_m>1/mnMinDepth && dp>0.25)  continue; // FIXME: Occlusion assumption

			cv::circle( imRGB, cvpoint, 1, color, -1, 8 );
		}

		std::cout <<  "done" << std::endl;

		cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );
	  	cv::imshow( "image", imRGB );
		cv::waitKey(.1);

		std::vector<int> compression_params;
    	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    	compression_params.push_back(1);
		cv::imwrite("texture_geometry.png", imRGB, compression_params);
	//}
}

