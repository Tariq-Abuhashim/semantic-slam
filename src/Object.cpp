
/*
 * @file Object.cpp
 * This is part of Semantic SLAM.
 * Functions to handle each object instance.
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-07-00
 */

#include "Object.hpp"
#include <string>
#include <unordered_map>

long unsigned int Object::nNextId=0;

Object::Object(KeyFrame* pKF, cv::Rect Box, cv::Mat imD, cv::Mat K, int Sensor) : 
	mdMinPointsCount(15), mnTracks(0), mnHeight(480), mnWidth(640), mpRefKF(pKF), 
	mnMinDepth(0.2), mnMaxDepth(30), mRed(RandomFloat()), mGreen(RandomFloat()), mBlue(RandomFloat()),
	mProbThd(0.75), mnDist(2), mnRes(0.5), mSensor(Sensor), mK(K) // lidar 1, RGBD 4
{
	mnId=nNextId++;
	const cv::Mat Twc = pKF->GetPoseInverse();
	
	// Define the TSDF reference KeyFrame pose
	std::vector<float> base2world;
	for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) 
	base2world.push_back(Twc.at<float>(r, c)); // here, first KF is used as base of TSDF

	// Define the TSDF origin
	//float z = (float)(depth_mat.at<unsigned short>(bbox.height+bbox.y, bbox.x)) / 5000.0f
	//float z = mnMinDepth;
	//float x = (Box.x            - mK.at<float>(0,2))*z*(1.0f/mK.at<float>(0,0)); 
	//float y = (Box.height+Box.y - mK.at<float>(1,2))*z*(1.0f/mK.at<float>(1,1));

	std::vector<float> origin(3,1000);
	for (int r = 0; r < imD.rows; ++r)
    for (int c = 0; c < imD.cols; ++c)
	{
		float z = imD.at<float>(r, c); // FIXME: this is depth, not z
		if(z<=0.0) continue;
		//cout << z << endl;
		float x = (c - mK.at<float>(0,2))*z*(1.0f/mK.at<float>(0,0)); 
		float y = (r - mK.at<float>(1,2))*z*(1.0f/mK.at<float>(1,1));
		origin[0] = min(x,origin[0]);
		origin[1] = min(y,origin[1]);
		origin[2] = min(z,origin[2]);
	}

	//cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
	//cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
    //cv::Mat twc = Twc.rowRange(0,3).col(3);mSensor
    //x3Dc = Rwc*x3Dc + twc;
	//std::vector<float> origin;
	//for (int i = 0; i < 3; ++i)
	//origin.push_back(x3Dc.at<float>(i));

	//cout << " Origin : " << endl;
	//cout << "    2D: " << Box.height+Box.y << " " << Box.x << endl;
	//cout << "    3D: ";
	//for (int i = 0; i < 3; ++i)
	//cout << origin[i] << " ";
	//cout << endl;

	// Initialise a TSDF
	//tsdf = new TSDF(mnHeight, mnWidth, mnId, base2world, origin); //cpp
	//tsdf = new TSDFfusion(); // python
}

Object::~Object()
{

	nNextId--;
	// delete the TSDF
	//delete(tsdf);

	// save the Object points
	//ofstream f;
    //f.open("points_"+to_string(mnId)+".txt");
    //f << fixed;
	//for(auto sit = mspObjectPoints.begin(); sit != mspObjectPoints.end(); sit++)
	//{
	//	cv::Mat x3Dc = (*sit)->GetWorldPos();
	//	f<<setprecision(6)<<x3Dc.at<float>(0)<<" "<<x3Dc.at<float>(1)<<" "<<x3Dc.at<float>(2)<<endl;
	//}
	//f.close();
}

void Object::SetObjectParams( float ProbThd, float MinPointsCount, int Height, int Width, 
	float MinDepth, float MaxDepth, float Dist, float Res) {
	
	unique_lock<mutex> lock(mMutexObject);

	mProbThd = ProbThd;
	mdMinPointsCount = MinPointsCount;
	mnHeight = Height;
	mnWidth = Width;
	mnMinDepth = MinDepth;
	mnMaxDepth = MaxDepth;
	mnDist = Dist;
	mnRes = Res;
}

void Object::SaveToFile(const std::string& ext)
{

	// open a file stream
	std::ofstream f;
	std::string filename = msLabel + ext;
    f.open(filename.c_str());
    f << fixed;
	
	// save label and score
	f << msLabel << " " << mnScore << std::endl;

	// reference frame
	const cv::Mat Twc = mpRefKF->GetPoseInverse();
	cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
	cv::Mat twc = Twc.rowRange(0,3).col(3);
	cv::Mat Rcw = Rwc.t();

	// save point coordinates wrt reference frame
	std::vector<ObjectPoint*> vObjectPoints=this->GetAllObjectPoints();
	for(size_t MPi=0; MPi<vObjectPoints.size(); MPi++)
	{
		ObjectPoint* MP = vObjectPoints[MPi];
		if(MP->isBad()) continue; // (0.75 lidar, 0.5 RGBD)
		cv::Mat x3Dc = MP->GetWorldPos();
		//x3Dc = Rcw*(x3Dc-twc);
		f << x3Dc.at<float>(0) << " " << x3Dc.at<float>(1) << " " << x3Dc.at<float>(2) << std::endl;
	}

	f.close();

}

float Object::RandomFloat()
{
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void Object::Integrate(cv::Mat imRGB, cv::Mat imD, cv::Mat cam2world)
{
	//int H = depth_mat.rows;
	//int W = depth_mat.cols;
	//float depth[H * W];
	//for (int r = 0; r < H; ++r)
    //for (int c = 0; c < W; ++c) 
	//{
	//	depth[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c));// / 5000.0f;
    //  if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
	//	depth[r * W + c] = 0;
	//}

	//std::vector<float> cam2world_vec;
	//for (int i = 0; i < 4; ++i) 
    //	cam2world_vec.insert(cam2world_vec.end(), cam2world.ptr<float>(i), cam2world.ptr<float>(i)+4);

	float* depth = (float*)imD.data;
	std::vector<float> cam2world_vec;
	cam2world_vec.insert(cam2world_vec.end(), (float*)cam2world.data, (float*)cam2world.data+16);
	
	tsdf->Integrate(depth, cam2world_vec); // cpp
	//tsdf->Integrate(imRGB, imD); // python
}

void Object::AddKeyFrame(KeyFrame* pKF)
{
	unique_lock<mutex> lock(mMutexObject);
    mspKeyFrames.insert(pKF); // set
	//mspKeyFrames.push_back(pKF); // vector
    //if(pKF->mnId>mnMaxKFid)
    //    mnMaxKFid=pKF->mnId;
}

void Object::AddObjectPoint(ObjectPoint* pMP) // MP (hence also links to mvKeys in KF)
{
    unique_lock<mutex> lock(mMutexObject);
	mspObjectPoints.insert(pMP); // set
	//mspObjectPoints.push_back(pMP); // vector
}

vector<KeyFrame*> Object::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexObject);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<ObjectPoint*> Object::GetAllObjectPoints()
{
    unique_lock<mutex> lock(mMutexObject);
    return vector<ObjectPoint*>(mspObjectPoints.begin(),mspObjectPoints.end());
}


KeyFrame* Object::GetReferenceKeyFrame()
{
	unique_lock<mutex> lock(mMutexObject);
	return mpRefKF;
}

// Keyframes observing the contour and associated index in keyframe
std::map<KeyFrame*,size_t> Object::GetObservations()
{
	unique_lock<mutex> lock(mMutexObject);
	return mObservations;
}

int Object::Observations() // returns number of contours
{
	unique_lock<mutex> lock(mMutexObject);
    return nObs;
}

void Object::AddObservation(KeyFrame* pKF, size_t idx) // KF + mvpContours
{
    unique_lock<mutex> lock(mMutexObject);
    if(mObservations.count(pKF)) // to prevent over-writing a measurement in std::map
        return;
    mObservations[pKF]=idx;
    nObs++;
}

void Object::EraseObservation(KeyFrame* pKF) // KF + mvContours
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexObject);
        if(mObservations.count(pKF))
        {
            mObservations.erase(pKF);
			nObs--;

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

void Object::AddContour(const Contour& cnt) // FIXME: should be in KeyFrame.cc
{
    unique_lock<mutex> lock(mMutexObject);
   	mvpContours.push_back(cnt);
}

Contour Object::GetContour(size_t idx) // FIXME: should be in KeyFrame.cc
{
    unique_lock<mutex> lock(mMutexObject);
   	return mvpContours[idx];
}

Contour Object::GetContour(KeyFrame* KF) // FIXME: should be in KeyFrame.cc
{
    unique_lock<mutex> lock(mMutexObject);
	size_t idx = mObservations[KF];
   	return mvpContours[idx];
}

size_t Object::GetNumberOfContours() // FIXME: should be in KeyFrame.cc
{
    unique_lock<mutex> lock(mMutexObject);
   	return mvpContours.size();
}

void Object::AddBoundingBox(const cv::Rect& box) // FIXME: should be in KeyFrame.cc
{
    unique_lock<mutex> lock(mMutexObject);
   	mvpBoundingBoxes.push_back(box);
}

void Object::SetLabel(const string& label) {
	unique_lock<mutex> lock(mMutexObject);
	msLabel = label;
}

std::string Object::GetLabel() {
	unique_lock<mutex> lock(mMutexObject);
	return msLabel;
}

void Object::UpdateScore(const double score) {
	unique_lock<mutex> lock(mMutexObject);
	mnScore = score;
}

double Object::GetScore() {
	unique_lock<mutex> lock(mMutexObject);
	return mnScore;
}

bool Object::IsInObject(ObjectPoint * pMP)
{
    unique_lock<mutex> lock(mMutexObject);
    return (mspObjectPoints.count(pMP));
}

bool Object::HasEnoughObjectPoints()
{
	unique_lock<mutex> lock(mMutexObject);
	return (mspObjectPoints.size()>mdMinPointsCount);
}

int Object::GetMaxObservations()
{
	int MaxObs = 0;
	for(auto sit = mspObjectPoints.begin(); sit != mspObjectPoints.end(); sit++)
	{
		int nObs = (*sit)->Observations();
		MaxObs = MaxObs + nObs; // nObs reflects how many times a ObjectPoint has agreed with this object contours
		//if(nObs > MaxObs)
		//	MaxObs = nObs;
	}
	return (MaxObs/2)/mspObjectPoints.size(); // FIXME: ObjectPoint->AddObservation() is incrementing 2 a time !!!!
}

cv::Point Object::ProjectObjectPoint(ObjectPoint* MP, KeyFrame* KF)
{
	//unique_lock<mutex> lock(mMutexPose);

	const float fx= mK.at<float>(0,0);
	const float fy= mK.at<float>(1,1);
	const float cx= mK.at<float>(0,2);
	const float cy= mK.at<float>(1,2);

	//std::cout << mK << std::endl;

	const cv::Mat x3Dc = MP->GetWorldPos();
	const cv::Mat Twc = KF->GetPoseInverse();

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

void Object::AddNewObjectPoints(Contour maskContour, const std::vector<std::vector<cv::Point> >& clusters, const cv::Mat& imDepth, KeyFrame* KF, float score)
{
	// Loop DoN clusters to resolve overlaping with the mask
	//std::vector <int> overlap(clusters.size(),0);
	for(int i=0; i<clusters.size(); i++)
	{
		std::vector<cv::Point> cluster = clusters[i];
		float count = 0;
		for(int j=0; j<cluster.size(); j++)
		{
			float dist = -100;
			
			for(size_t Ci = 0; Ci<maskContour.size(); Ci++)
			{
				dist = pointPolygonTest(maskContour[Ci], cluster[j], true);	
				if(dist>mnDist) 
				{
					//std::cout << "Oii" << std::endl; 
					count++;
				}
			}
			//if(count>0) {overlap[i]++;} // count this cvpoint in
		}
		float overlap = count/cluster.size();
		//std::cout <<  i << " " << cluster.size() << " " << count << " " << overlap << std::endl;
		if(overlap>0.5f)
		{
			AddSegment(cluster, imDepth, KF, score);
		}
	}
}
		
void Object::AddNewObjectPoints(Contour maskContour, const cv::Mat& imDepth, KeyFrame* KF, float score)
{

	const cv::Mat Twc = KF->GetPoseInverse();
	cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
	cv::Mat twc = Twc.rowRange(0,3).col(3);
	cv::Mat Rcw = Rwc.t();

	//bool HasPoint = false;
	//int count = 0;
	//for(size_t Ci = 0; Ci< maskContour.size(); Ci++)
	//{
	//	if(mnTracks > 0.85*cv::contourArea(maskContour[Ci]))
	//	{
	//		count++;
	//	}
	//}
	//if(count==maskContour.size()) return;

	bool runKdtree = false;
	if (mspObjectPoints.size()==0) runKdtree = false;
	vector<cv::Point> features;
	if (runKdtree)
	{	
		for(auto sit = mspObjectPoints.begin(); sit != mspObjectPoints.end(); sit++)
		{

			cv::Point cvpoint = ProjectObjectPoint(*sit, KF);

			if(cvpoint.x>5 && cvpoint.x<imDepth.cols-5 && cvpoint.y>5 && cvpoint.y<imDepth.rows-5)
			{
				cv::Mat x3Dc = (*sit)->GetWorldPos();
				x3Dc = Rcw*(x3Dc-twc);
				float x,y,z;
				x = x3Dc.at<float>(0);
				y = x3Dc.at<float>(1);
				z = x3Dc.at<float>(2);
				float p_hat = 1/std::sqrt(x*x+y*y+z*z);
				if (p_hat>1/mnMinDepth || p_hat<1/mnMaxDepth) 
					continue; // look at ranges between mnMinDepth and mnMaxDepth

				float p_m = 1/imDepth.at<float>(cvpoint.y, cvpoint.x);
				float dp = (p_hat-p_m)/abs(p_hat-p_m); // FIXME: is this a good measure ?
				if (p_m<1/mnMaxDepth && p_m>1/mnMinDepth && dp>0.25)  
					continue; // FIXME: Occlusion assumption is weak

				for(size_t Ci = 0; Ci<maskContour.size(); Ci++)
				{
					float dist = pointPolygonTest(maskContour[Ci], cvpoint, true);
					if (dist>mnDist) // FIXME: Occlusion assumption is weak
					{
						features.push_back(cvpoint); // only consider points inside contours and visible in KeyFrame
						int idx = round(cvpoint.x)*imDepth.rows + round(cvpoint.y);
						(*sit)->AddObservation(KF,idx); // count to the number of times a ObjectPoint has been seen
					}
				}
			}
		}	
	}
	else
	{
		features.push_back(cv::Point2f(0.0,0.0));
	}

	// KdTree with 10 random trees
    cv::Mat nodes = cv::Mat(features).reshape(1);
	if(nodes.type()!=CV_32F) { nodes.convertTo(nodes, CV_32F); }
    cv::flann::KDTreeIndexParams indexParams(10);
    cv::flann::Index kdtree(nodes, indexParams);

	int count = 0;

	// Add more ObjectPoints
	//if(!ObjFound)
	//{
		//cv::Rect bbox = boxes[mi];
		//for(int col=bbox.x; col<bbox.width+bbox.x; col++)
		for(int col=0; col<imDepth.cols; col+=1) // FIXME: don't search the whole image frame
		{
			//for(int row=bbox.y; row<bbox.height+bbox.y; row++)
			for(int row=0; row<imDepth.rows; row+=1) // FIXME: don't search the whole image frame
			{
				const float range = imDepth.at<float>(row, col);
				//if (z) cout << z << " ";
				if (range<mnMinDepth || range>mnMaxDepth) continue;

				float dist = -100;
				for(size_t Ci = 0; Ci<maskContour.size(); Ci++)
				{
					dist = pointPolygonTest(maskContour[Ci], cv::Point2f(col,row), true);
					
					if (dist>mnDist)
					{

						if (runKdtree)
						{
							unsigned int num_neighbours = 32;
							unsigned int num_searches = 64;
							vector<float> query;
							query.push_back(col);
							query.push_back(row);
							vector<int> indices(num_neighbours);
							vector<float> dists(num_neighbours);
	  						kdtree.knnSearch(query, indices, dists, num_neighbours, 
																cv::flann::SearchParams(num_searches));

							//cout<< dists[0] << endl;
							//cout<< features[indices[0]].x<< " "<< features[indices[0]].y << endl;					
							//cout << col << " " << row << endl;
							//waitKey(0);

							if(dists[0]<mnRes) continue; // take point samples at this resolution
						}

						//Is there a point close? maybe don't add a new point then
						//FIXME: Use faster lookup, maybe Kdtree ?
						//bool HasPoint = false;
						//vector<ObjectPoint*> vObjectPoints = MO->GetAllObjectPoints();
						//for(size_t ii=0; ii<vObjectPoints.size(); ii++)
						//{
						//	Point cvpoint = ProjectObjectPoint(vObjectPoints[ii], KF, K);
						//	if (sqrt(pow(cvpoint.x-col,2)+pow(cvpoint.y-row,2))<5)
						//	{
						//		HasPoint = true;
						//		break;
						//	}
						//}
						//if(HasPoint) continue;


						// Undistort points
						cv::Mat mat(1,2,CV_32F);
						mat.at<float>(0)=row;
						mat.at<float>(1)=col;
						mat=mat.reshape(2);
						undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
						mat=mat.reshape(1);

						// reproject points
						float v = mat.at<float>(0);
						float u = mat.at<float>(1);
						float x = (u-mK.at<float>(0,2))/mK.at<float>(0,0);
						float y = (v-mK.at<float>(1,2))/mK.at<float>(1,1);

						float d;
						if(mSensor==1)
						{
							float rim = sqrt(x*x + y*y + 1);
				 			d = range/rim; // FIXME: Lidar
						}
						else if(mSensor==2)
							d = range; // FIXME: RGBD

						cv::Mat x3Dc = (cv::Mat_<float>(3,1) << d*x, d*y, d);
						cv::Mat Twc = KF->GetPoseInverse();
						cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
						cv::Mat twc = Twc.rowRange(0,3).col(3);
						x3Dc = Rwc*x3Dc+twc;
						ObjectPoint* MP = new ObjectPoint(x3Dc, KF);
						int idx = col*imDepth.rows + row;
						MP->AddObservation(KF,idx);

						//MP->mbTrackInView=true;//FIXME: flag to indicate q point has already been seen by an object
						MP->UpdateForegroundProbability(score);
						MP->SetProbabilityThreshold(mProbThd);

						AddObjectPoint(MP);

						count++;
					}
				}
			}
		}
	//}

	//cout << " (" << count << ") " ;
}

void Object::AddSegment(Contour maskContour, const cv::Mat& imDepth, KeyFrame* KF, float score)
{

	const cv::Mat Twc = KF->GetPose();
	cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
	cv::Mat twc = Twc.rowRange(0,3).col(3);
	cv::Mat Rcw = Rwc.t();

	// Kd-tree
	std::vector<cv::Point> features = this->Get2dFeatures(KF, imDepth);
	cv::Mat nodes = cv::Mat(features).reshape(1);
	if(nodes.type()!=CV_32F) { nodes.convertTo(nodes, CV_32F); }
    cv::flann::KDTreeIndexParams indexParams(10);
    cv::flann::Index kdtree(nodes, indexParams);


	//cv::Rect bbox = boxes[mi];
	//for(int col=bbox.x; col<bbox.width+bbox.x; col++)
	for(int col=0; col<imDepth.cols; col+=1) // FIXME: don't search the whole image frame
	{
		//for(int row=bbox.y; row<bbox.height+bbox.y; row++)
		for(int row=0; row<imDepth.rows; row+=1) // FIXME: don't search the whole image frame
		{
			float range = imDepth.at<float>(row, col);
			if (range<mnMinDepth || range>mnMaxDepth) continue;

			float dist = -100;
			dist = pointPolygonTest(maskContour[0], cv::Point2f(col,row), true);
			if (dist<mnDist) continue;

			// use kdtree to keep acceptable 3D resolution
			unsigned int num_neighbours = 32;
			unsigned int num_searches = 64;
			vector<float> query;
			query.push_back(col);
			query.push_back(row);
			vector<int> indices(num_neighbours);
			vector<float> dists(num_neighbours);
			kdtree.knnSearch(query, indices, dists, num_neighbours, cv::flann::SearchParams(num_searches));
			if(dists[0]<mnRes) continue; // FIXME: Object resolution : take point samples at this resolution (lidar: 1, RGBD: 4)

			// Undistort points
			cv::Mat mat(1,2,CV_32F);
			mat.at<float>(0)=row;
			mat.at<float>(1)=col;
			mat=mat.reshape(2);
			undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
			mat=mat.reshape(1);

			// Reproject points
			float v = mat.at<float>(0);
			float u = mat.at<float>(1);
			float x = (u-mK.at<float>(0,2))/mK.at<float>(0,0);
			float y = (v-mK.at<float>(1,2))/mK.at<float>(1,1);
			
			float d;
			if(mSensor==1)
			{
				float rim = sqrt(x*x + y*y + 1);
	 			d = range/rim; // FIXME: Lidar
			}
			else if(mSensor==2)
				d = range; // FIXME: RGBD

			cv::Mat x3Dc = (cv::Mat_<float>(3,1) << d*x, d*y, d);
			cv::Mat Twc = KF->GetPoseInverse();
			cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
			cv::Mat twc = Twc.rowRange(0,3).col(3);
			x3Dc = Rwc*x3Dc+twc;

			// add new MapPoint
			ObjectPoint* MP = new ObjectPoint(x3Dc, KF);
			int idx = col*imDepth.rows + row;
			MP->AddObservation(KF,idx);
			MP->UpdateForegroundProbability(score);
			AddObjectPoint(MP);
		}

	}

}

std::vector<cv::Point> Object::Get2dFeatures(KeyFrame* KF, const cv::Mat& imDepth)
{
	std::vector<cv::Point> features;

	const cv::Mat Twc = KF->GetPoseInverse();
	cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
	cv::Mat twc = Twc.rowRange(0,3).col(3);
	cv::Mat Rcw = Rwc.t();

	if (mspObjectPoints.size()==0) 
	{
		features.push_back(cv::Point2f(0.0,0.0));
		return features;
	}

	for(auto sit = mspObjectPoints.begin(); sit != mspObjectPoints.end(); sit++)
	{
		cv::Point cvpoint = ProjectObjectPoint(*sit, KF);
		if(cvpoint.x>5 && cvpoint.x<imDepth.cols-5 && cvpoint.y>5 && cvpoint.y<imDepth.rows-5)
		{
			cv::Mat x3Dc = (*sit)->GetWorldPos();
			x3Dc = Rcw*(x3Dc-twc);
			float x,y,z;
			x = x3Dc.at<float>(0);
			y = x3Dc.at<float>(1);
			z = x3Dc.at<float>(2);
			float p_hat = 1/std::sqrt(x*x+y*y+z*z);
			if (p_hat>1/mnMinDepth || p_hat<1/mnMaxDepth) continue; // look at ranges between mnMinDepth and mnMaxDepth

			float p_m = 1/imDepth.at<float>(cvpoint.y, cvpoint.x);
			float dp = (p_hat-p_m)/abs(p_hat-p_m); // FIXME: is this a good measure ?
			if (p_m<1/mnMaxDepth && p_m>1/mnMinDepth && dp>0.25) continue; // FIXME: Occlusion assumption is weak

			features.push_back(cvpoint); // only consider points inside contours and visible in KeyFrame

		}
	}

	return features;
}

void Object::AddSegment(const std::vector<cv::Point>& cluster, const cv::Mat& imDepth, KeyFrame* KF, float score)
{
	const cv::Mat Twc = KF->GetPoseInverse();
	cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
	cv::Mat twc = Twc.rowRange(0,3).col(3);
	cv::Mat Rcw = Rwc.t();

	// insert the segment into mvpSegments
	mvpSegments.push_back(cluster);

	// Kd-tree
	std::vector<cv::Point> features = this->Get2dFeatures(KF, imDepth);
	cv::Mat nodes = cv::Mat(features).reshape(1);
	if(nodes.type()!=CV_32F) { nodes.convertTo(nodes, CV_32F); }
    cv::flann::KDTreeIndexParams indexParams(10);
    cv::flann::Index kdtree(nodes, indexParams);

	// insert the segment points into mspObjectPoints
	for(int j=0; j<cluster.size(); j++)
	{
		cv::Point cvpoint = cluster[j];
		int col = cvpoint.x;
		int row = cvpoint.y;
		float range = imDepth.at<float>(row, col);
		if (range<mnMinDepth || range>mnMaxDepth) continue;

		// use kdtree to keep acceptable 3D resolution
		unsigned int num_neighbours = 32;
		unsigned int num_searches = 64;
		vector<float> query;
		query.push_back(col);
		query.push_back(row);
		vector<int> indices(num_neighbours);
		vector<float> dists(num_neighbours);
		kdtree.knnSearch(query, indices, dists, num_neighbours, cv::flann::SearchParams(num_searches));
		if(dists[0]<mnRes) continue; //Object resolution: take point samples at this resolution (lidar: 1, RGBD: 4)

		// Undistort points
		cv::Mat mat(1,2,CV_32F);
		mat.at<float>(0)=row;
		mat.at<float>(1)=col;
		mat=mat.reshape(2);
		undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
		mat=mat.reshape(1);

		// Reproject points
		float v = mat.at<float>(0);
		float u = mat.at<float>(1);
		float x = (u-mK.at<float>(0,2))/mK.at<float>(0,0);
		float y = (v-mK.at<float>(1,2))/mK.at<float>(1,1);
		
 		float d;
		if(mSensor==1)
		{
			float rim = sqrt(x*x + y*y + 1);
	 		d = range/rim; // FIXME: Lidar
		}
		else if(mSensor==2)
			d = range; // FIXME: RGBD

		cv::Mat x3Dc = (cv::Mat_<float>(3,1) << d*x, d*y, d);
		cv::Mat Twc = KF->GetPoseInverse();
		cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
		cv::Mat twc = Twc.rowRange(0,3).col(3);
		x3Dc = Rwc*x3Dc+twc;

		// add new MapPoint
		ObjectPoint* MP = new ObjectPoint(x3Dc, KF);
		int idx = col*imDepth.rows + row;
		MP->AddObservation(KF,idx);
		MP->UpdateForegroundProbability(score);
		AddObjectPoint(MP);
	}

}

void Object::CheckObjectPoints(KeyFrame* KF2)
{
	int MaxObs = GetMaxObservations();
	std::cout << "MaxObs: " << MaxObs << std::endl;
	if(MaxObs<3) return; // point has not been seen often enough

	for(auto sit = mspObjectPoints.begin(); sit != mspObjectPoints.end(); sit++)
	{
		map<KeyFrame*, size_t> Obs = (*sit)->GetObservations();

		std::vector<KeyFrame*> vKeyFrames;
		for(auto mit = Obs.begin(); mit != Obs.end(); mit++)
			vKeyFrames.push_back(mit->first);
		sort(vKeyFrames.begin(),vKeyFrames.end(),KeyFrame::lId);

		KeyFrame* KF1 = vKeyFrames.back();
		if((KF2->mnId-KF1->mnId)<4) continue; // too early, only few keyframes

		int nObs = (*sit)->Observations();
		if(nObs<MaxObs) (*sit)->mbTrackInView=false; //mspObjectPoints.erase(sit, mspObjectPoints.end());
	}
}

void Object::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        //unique_lock<mutex> lock1(mMutexFeatures);
        //unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        //KeyFrame* pKF = mit->first;
        //pKF->EraseObjectPointMatch(mit->second); // FIXME: not implemented for contours yet
    }

    //mpMap->EraseObjectPoint(this); // FIXME: not implemented for contours yet
}

bool Object::isBad() 
{
	return mbBad;
}

