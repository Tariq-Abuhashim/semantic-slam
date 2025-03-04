

/*
*
* Tariq Abuhashim
* t.abuhashim@gmail.com
* July, 2019
*
*/

#include "MaskRCNN.hpp"
#include "MapObject.hpp"
#include "cv.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <set>

//#include "Thirdparty/ORB_SLAM2/include/System.h"
#include "Thirdparty/ORB_SLAM2/include/KeyFrame.h"
#include "Thirdparty/ORB_SLAM2/include/MapPoint.h"
#include "Thirdparty/ORB_SLAM2/include/Map.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM2;

void LoadCategories(unordered_map<int,string> &categories, 
					const string &strCategoriesFilename);
void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD);
void LoadMap(const string &BundlerFilename, Map* mpMap, KeyFrameDatabase *KFDB);

cv::Point ProjectMapPoint(MapPoint* MP, KeyFrame* KF)
{
	//unique_lock<mutex> lock(mMutexPose);

	const float fx= 535.4;
	const float fy= 539.2;
	const float cx= 320.1;
	const float cy= 247.6;

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


int main( int argc, char** argv ) {

	if( argc != 2) {
		cout <<" Usage: label_instance /path/to/result_and_config/folder" << endl;
		return -1;
    }

	cout << endl << "-------" << endl;
	cout << " * Reading categories..." << endl;
	unordered_map<int,std::string> categories;
	//string strCategoriesFilename = string(string(argv[1])+"/config/categories.txt");
	string strCategoriesFilename = "../config/categories.txt";
	LoadCategories(categories, strCategoriesFilename);
	if(categories.empty()) {
		cerr << endl << "Empty categories at: " << strCategoriesFilename << endl;
		return 1;
	}
	cout << " * Categories loaded!" << endl;

	//Load ORB Vocabulary
    cout << " * Loading ORB Vocabulary. This could take a while..." << endl;
	string strVocFilename = "../config/ORBvoc.txt";
    ORBVocabulary* mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFilename);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFilename << endl;
        exit(-1);
    }
    cout << " * Vocabulary loaded!" << endl;

    //Create KeyFrame Database
    KeyFrameDatabase *KFDB = new KeyFrameDatabase(*mpVocabulary);

	// Rebuild the ORB_SLAM2 mpMap
	cout << " * Loading Map from bundler file..." << endl;
	//string strBundleFilename = string(string(argv[1])+"/result/bundle.txt");
	string strBundleFilename = "../result/bundle.txt";
	Map* mpMap = new Map();
	LoadMap(strBundleFilename, mpMap, KFDB); // Loads map from file, this also includes pointers to all keyframes (which include 2d points), pointers to all MapPoints (3d points)
	cout << " * Map Loaded!" << endl;

	vector<KeyFrame*> AllKeyFrames;
	vector<MapPoint*> AllMapPoints;
	AllKeyFrames = mpMap->GetAllKeyFrames();
	sort(AllKeyFrames.begin(),AllKeyFrames.end(),KeyFrame::lId);
	AllMapPoints = mpMap->GetAllMapPoints();

	// Retrieve paths to KeyFrame images
	cout << " * Loading image paths..." << endl;
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    //string strAssociationFilename = string(string(argv[1])+"/result/associations.txt"); // custom file relating KeyFrames RGB to D images
	string strAssociationFilename = "../result/associations.txt";
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD);
	cout << " * Image names Loaded!" << endl;

	// Check consistency in the number of images and depthmaps
    //int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty()) {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size()) {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

	size_t nKFs = AllKeyFrames.size();
	size_t nMPs = AllMapPoints.size();

	cout << endl << "-------" << endl;
    cout << " * Images in the sequence: " << nKFs << endl;
	cout << " * 3D points in the map: " << nMPs << endl;
	cout << " * Labelled categories: " << categories.size() << endl;

/*
	vector<vector<KeyPoint> > vKeyPoints;
	vKeyPoints.reserve(nKFs);
	vKeyPoints.resize(nKFs);
	for(size_t MPi=0; MPi<nMPs; MPi++)
	{
		MapPoint* MP = AllMapPoints[MPi];
		map<KeyFrame*, size_t> Obs = MP->GetObservations();
		for(auto mit = Obs.begin(); mit != Obs.end(); mit++)
		{
			KeyFrame* KF = mit->first;
			size_t idx = mit->second;
			cout << KF->mnId << " , " << idx << " , " << (KF->mvKeys).size() << endl;
			
			vKeyPoints[KF->mnId].push_back(KF->mvKeys[idx]);
		}
	}
	cout << " * vKeyPoints.size(): " << vKeyPoints.size() << endl;

	for(size_t i=0;i<vKeyPoints.size(); i++)
	{
		Mat imRGB;
        imRGB = imread(string(argv[1]) + "/" + vstrImageFilenamesRGB[i], CV_LOAD_IMAGE_UNCHANGED);

		cout << " * vKeyPoints[i].size(): " << vKeyPoints[i].size() << endl;

		for(size_t j=0;j<vKeyPoints[i].size(); j++)
		{
			cv::KeyPoint KP = vKeyPoints[i][j];
			circle( imRGB, Point(KP.pt.x,KP.pt.y), 1, Scalar( 0, 0, 255 ), -1, 8 );
		}
		cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );
	  	cv::imshow( "image", imRGB );
		waitKey(0);
	}

	//return 0;
*/
	cout << endl << "-------" << endl;
	MaskRCNN* mMaskRCNN = new MaskRCNN();
	unordered_set<MapObject*> Objects; // this should be in mpMap, and should be updated using [ mpMap->AddMapObject(MO) ]
	std::vector<std::vector<Contour> > vKFsContours; // All object contours  in KeyFrames. 
	// FIXME: Each KeyFrame should have [ std::vector<cv::contours> ] in KeyFrame.h (similar to mvKeys). 
	// It is then linked to MapObject using mObservations
	for(size_t KFi=0; KFi<nKFs; KFi++) 
	{

		KeyFrame *KF = AllKeyFrames[KFi];
		
		Mat imRGB;
        imRGB = imread(string(argv[1]) + "/" + vstrImageFilenamesRGB[KFi], CV_LOAD_IMAGE_UNCHANGED);
        if(imRGB.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(argv[1]) << "/" << vstrImageFilenamesRGB[KFi] << endl;
            return 1;
        }
		
		// Run mask-rcnn
		cout << " -- Mask-RCNN Run() ";
		vector<Rect> boxes;
		vector<Mat> masks;
		vector<int> labels;
		vector<double> scores;
		mMaskRCNN->Run(imRGB, boxes, masks, labels, scores);
		//cout << " -- Class labels: ";
		//for(vector<int>::iterator itr=labels.begin(); itr<labels.end(); itr++) 
		//	cout << categories[*itr] << ", ";
		//cout << endl;

		MapObject *MO;
		for(size_t mi=0; mi<masks.size(); mi++) 
		{

			// Find the contours in the mask
			Contour maskContour;
			vector<cv::Vec4i> hierarchy;
			cv::findContours( masks[mi], maskContour, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

			// Check if this contour belongs to a previous MapObject by back-projecting Objects MapPoints
			bool ObjFound = false;
			for(auto sit=Objects.begin(); sit!=Objects.end(); sit++)
			{
				MO = *sit;
				vector<MapPoint*> vMapPoints = MO->GetAllMapPoints();
				int count = 0;
				for(size_t MPi=0; MPi<vMapPoints.size(); MPi++)
				{
					MapPoint* MP = vMapPoints[MPi];
					//cv::Point ImagePoint = ProjectMapPoint(MP, KF);
					if(MP->IsInKeyFrame(KF))
					//if(ImagePoint.x>0&&ImagePoint.x<imRGB.cols&&ImagePoint.y>0&&ImagePoint.y<imRGB.rows)
					{
						int idx = MP->GetIndexInKeyFrame(KF);
						cv::KeyPoint KP = KF->mvKeys[idx];
						float dist = -100;
						for(size_t Ci = 0; Ci< maskContour.size(); Ci++)
						{
							dist = cv::pointPolygonTest(maskContour[Ci], cv::Point2f(KP.pt.x,KP.pt.y), true);
							//dist = cv::pointPolygonTest(maskContour[Ci], ImagePoint, true);
							//std::cout << dist << std::endl;
							if (dist>=0) break;
						}
						if(dist>=0)
						{
							count++;
							if (count>1)
							{
								ObjFound = true;
								break;
							}
						}
					}
				}
				if(ObjFound) break;
			}

			// Create a new MapObject (MO)
			if(!ObjFound)
			{
				MO = new MapObject(KF, boxes[mi], );
				MO->SetLabel(categories[labels[mi]]); // FIXME: Label needs to be incrementally updated  ;)
			}

			// Add contour observation to either old or new MO
			MO->AddKeyFrame(KF);
			MO->AddObservation(KF,mi); // Contour with index (mi) in KeyFrame (KF), FIXME: currently useless because contours are pushed at the same order as KeyFrames
			MO->AddContour(maskContour); // FIXME: should should be implemented as [ KF->AddContour(maskContour) ]
			MO->AddBoundingBox(boxes[mi]); // FIXME: should should be implemented as [ KF->AddBoundingBox(boxes[mi]) ]

			// Add new MapPoints to MO
			//vector<MapPoint*> vMP = KF->GetMapPointMatches(); // FIXME: THIS IS EMPTY !!!!!!!!!!!!
			for(size_t MPi=0; MPi<nMPs; MPi++) // FIXME: should be looping FKMapPoints which reads from KeyFrame protected, I can not populate
			{
				MapPoint* MP = AllMapPoints[MPi];

				//MapPoint* MP = vMP[MPi];
				//cv::Point ImagePoint = ProjectMapPoint(MP, KF);
				if(MP->IsInKeyFrame(KF))
				//if(ImagePoint.x>0&&ImagePoint.x<imRGB.cols&&ImagePoint.y>0&&ImagePoint.y<imRGB.rows)
				{
					int idx = MP->GetIndexInKeyFrame(KF);
					cv::KeyPoint KP = KF->mvKeys[idx];
					float dist = -100;
					for( size_t Ci = 0; Ci< maskContour.size(); Ci++ )
					{
						dist = cv::pointPolygonTest(maskContour[Ci], cv::Point2f(KP.pt.x,KP.pt.y), true);
						//dist = cv::pointPolygonTest(maskContour[Ci], ImagePoint, true);
						if (dist>=0)
						{
							break;
						}
					}
					
					if(dist>=0 && !MO->IsInMapObject(MP))
					{
						std::cout << dist << std::endl;
						MO->AddMapPoint(MP); // this link MapPoint pointer and hence mvKeys index too in KF
					}
				
				}
			}

			// Insert object ?
			vector<MapPoint*> vMapPoints = MO->GetAllMapPoints();
			if(!ObjFound && MO->HasEnoughMapPoints())
			{
				Objects.insert(MO); // if its a new Object insert to mpMap
				cout << " [Mask " << mi << "] contains a new object: " << MO->GetLabel() << " with " << vMapPoints.size() << " MapPoints." << endl;
			}
			else if (MO->HasEnoughMapPoints())
			{
				cout << " [Mask " << mi << "] contains an old object: " << MO->GetLabel() << " with " << vMapPoints.size() << " MapPoints." << endl;
			}

			// Draw contours, labels, MapPoints
			if(true)
			{
	  			RNG rng(12345);
				Scalar color = Scalar( 0, 255, 0 ); // Green = Old Object
				for( size_t Ci = 0; Ci< maskContour.size(); Ci++ )
			 	{
					if(!ObjFound && MO->HasEnoughMapPoints())
			   			color = Scalar( 0, 0, 255 ); // Red = New Object

					if (MO->HasEnoughMapPoints())
					{
			   			drawContours( imRGB, maskContour, Ci, color, 1, 4, hierarchy, 0, Point() );
						Rect bbox = boxes[mi];
						putText(imRGB, MO->GetLabel(), Point(bbox.y,bbox.x), FONT_HERSHEY_DUPLEX, 0.5, color, 2);
					}
			 	}
				if (MO->HasEnoughMapPoints())
				{
					vector<MapPoint*> vMapPoints = MO->GetAllMapPoints();

					for(size_t MPi=0; MPi<vMapPoints.size(); MPi++)
					{
						MapPoint* MP = vMapPoints[MPi];
						//cv::Point ImagePoint = ProjectMapPoint(MP, KF);
						if(MP->IsInKeyFrame(KF))
						//if(ImagePoint.x>0&&ImagePoint.x<imRGB.cols&&ImagePoint.y>0&&ImagePoint.y<imRGB.rows)
						{
							int idx = MP->GetIndexInKeyFrame(KF);
							cv::KeyPoint KP = KF->mvKeys[idx];
							circle( imRGB, Point(KP.pt.x,KP.pt.y), 1, color, -1, 8 );
							//circle( imRGB, Point(ImagePoint.x,ImagePoint.y), 1, color, -1, 8 );
						}
					}
					cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );
	  				cv::imshow( "image", imRGB );
					waitKey(0);
				}
			}
	
		}

		cout << "Total number of MapPoints is " << AllMapPoints.size() << endl;
		cout << "Total number of objects in the map is " << Objects.size() << endl;

	}

	delete(mMaskRCNN);

	// Get 3D depth points for each object in KeyFrames
/*
	for(auto sit=Objects.begin(); sit!=Objects.end(); sit++)
	{
		MapObject* MO = *sit;
		std::vector<KeyFrame*> AllKeyFrames = MO->GetAllKeyFrames();
		sort(AllKeyFrames.begin(),AllKeyFrames.end(),KeyFrame::lId);
		for(size_t ci=0; ci<MO->GetNumberOfContours(); ci++)
		{
			Contour CurrentContour = MO->GetContour(ci);
			KeyFrame* KF = AllKeyFrames[ci];
			cv::Mat imRGB = imread(string(argv[1]) + "/" + vstrImageFilenamesRGB[KF->mnId], CV_LOAD_IMAGE_UNCHANGED);
			cv::Mat imD = imread(string(argv[1]) + "/" + vstrImageFilenamesD[KF->mnId], CV_LOAD_IMAGE_UNCHANGED);
			cv::namedWindow( "RGB", CV_WINDOW_AUTOSIZE );
			cv::namedWindow( "D", CV_WINDOW_AUTOSIZE );
  			cv::imshow( "RGB", imRGB );
			cv::imshow( "D", imD );
			waitKey(0);
			
		}
	}
*/

	
		
	return 0;

}

void LoadCategories(unordered_map<int,std::string> &categories, 
					const string &strCategoriesFilename) {
	ifstream fCategories;
  	fCategories.open(strCategoriesFilename.c_str());
	if(fCategories.is_open())
	{
    	while(!fCategories.eof()) {
			string line;
			getline(fCategories,line);
			if(!line.empty()) {
            	stringstream ss;
            	ss << line;
            	string value;
            	int key;
            	ss >> value;
            	ss >> key;
				//cout << key << " " << value << endl;
            	categories.insert(make_pair<int,string>((int)key,(string)value));
        	}
		}
		fCategories.close();
	}
	else
	{
		cerr << endl << "Failed to load categories at: " << strCategoriesFilename << endl;
	}

}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD) {
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
	if(!fAssociation.is_open())
	{
		cerr << endl << "Failed to load associations file at: " << strAssociationFilename << endl;
		return;
	}
    while(!fAssociation.eof()) {
        string s;
        getline(fAssociation,s);
        if(!s.empty()) {
            stringstream ss;
			ss << s;
			float t;
			string sRGB, sD;
			ss >> t;
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
			ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}

void LoadMap(const string &strBundlerFilename, Map* mpMap, KeyFrameDatabase *KFDB) { 
	
	ifstream fBundler;
    fBundler.open(strBundlerFilename.c_str());
	if(!fBundler.is_open())
	{
		cerr << endl << "Failed to load bundler file at: " << strBundlerFilename << endl;
		return;
	}
	ifstream fCoords;
	fCoords.open ("/home/mrt/Dev/semantic_slam_b/result/coords.txt");
	if (!fCoords.is_open())
	{
		cerr << endl << "Failed to load coords file at: " << strBundlerFilename << endl;
		return;
	}

	string s;
		
	size_t m, n; // sizes (KeyFrames, MapPoints)
	getline(fBundler,s);
	if(!s.empty())
	{
		stringstream ss;
		ss << s;
		ss >> m;
		ss >> n;
	}

	for(size_t i=0; i<m; i++)
	{
		cv::Mat Rcw(3,3,CV_32F);//
		cv::Mat tcw(3,1,CV_32F);//

		getline(fBundler,s); // Intrinsics

		for(int j=0; j<3; j++) { // Three lines 
			getline(fBundler,s); // Rotation - rows
			if(!s.empty()) {
				stringstream ss;
				for(int k=0; k<3; k++) {
					ss << s;  
					ss >> Rcw.at<float>(j,k);
				}
			}
		}

		getline(fBundler,s); // One line
		if(!s.empty()) {  // Translation
			stringstream ss;
			for(int j=0; j<3; j++) {
				ss << s;  
				ss >> tcw.at<float>(j);
			}
		}

		cv::Mat Tcw;
		Tcw = cv::Mat::eye(4,4,CV_32F);
		Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
		tcw.copyTo(Tcw.rowRange(0,3).col(3));

		// Create a Frame
		Frame CurrentFrame;
		CurrentFrame.mnId = i; // This affects KF->mnFrameId
		CurrentFrame.SetPose(Tcw);

		getline(fCoords, s);
		size_t camId;
		size_t index = s.find("#index = "); // first occurance
		if (index != string::npos) 
		{
			size_t comma = s.find(",", index+1);
			if (comma != string::npos) 
			{
				string ncam(s,index+9,comma-(index+9));
				stringstream ss(ncam);
				ss >> camId;
			}
		}
	
		assert(camId!=i); // FIXME: Why not working ?

		size_t npts;
		size_t keys = s.find("keys = "); // first occurance
		if (keys != string::npos) 
		{
			size_t comma = s.find(",", keys+1);
			if (comma != string::npos) 
			{
				string npts_str(s,keys+7,comma-(keys+7));
				stringstream ss(npts_str);
				ss >> npts;
			}
		}
		for(size_t i=0; i<npts; i++)
		{
			getline(fCoords, s);
			stringstream ss;
			ss << s;
			size_t KPId;
			ss >> KPId;
			float x, y;
			ss >> x;
			ss >> y;
			cv::KeyPoint *KP = new cv::KeyPoint();
			KP->pt.x = x;
			KP->pt.y = y;
			CurrentFrame.mvKeys.push_back(*KP);
			CurrentFrame.mvuRight.push_back(-1); // no right image - Monocular similar
			//cout << KPId << " " << x << " " << y << endl;
		}
    	
			
		// Create a KeyFrame
		KeyFrame* KF = new KeyFrame(CurrentFrame, mpMap, KFDB); // This affects KF->mnId
		//cout << "KeyFrame " << KF->mnId << " has " << KF->mvKeys.size() << " measurements." << endl;

		// Insert the KeyFrame in the map
		mpMap->AddKeyFrame(KF);  // Inserts the pointer into [ set<KeyFrame*> mspKeyFrames ]
		mpMap->mvpKeyFrameOrigins.push_back(KF);

		if (i==0) { }// check Tracking::StereoInitialization() to know what ORB_SLAM2::Map needs for initialisation
	}

		vector<KeyFrame*> AllKeyFrames;
		AllKeyFrames = mpMap->GetAllKeyFrames();

		for(size_t i=0; i<n; i++) 
		{
			getline(fBundler,s); // 3D MapPoint	
			cv::Mat x3D;
			if(!s.empty()) 
			{
				stringstream ss;
				ss << s;
				float x, y, z;
				ss >> x;
				ss >> y;
				ss >> z;
				x3D = (cv::Mat_<float>(3,1) << x, y, z);
				//cout << x3D << endl;
			}
			getline(fBundler,s); // Color
			getline(fBundler,s); // Observations
			MapPoint* MP;
        	if(!s.empty()) 
			{
				stringstream ss;
				ss << s;
				size_t k;
				ss >> k; // number of measurements
				for(size_t j=0; j<k; j++) 
				{
					size_t KFId; // KeyFrame Id
					ss >> KFId;
					KeyFrame* KF = AllKeyFrames[KFId];
					//cout << "KFId:"<<KFId << ", " << "KF->mnId:"<<KF->mnId << " " << "KF->mnFrameId:"<<KF->mnFrameId << endl;
					if(j==0) {
						MP = new MapPoint(x3D, KF, mpMap); // FIXME: assuming the first KeyFrame in the line is the reference of this MapPoint
						mpMap->AddMapPoint(MP); // Inserts the pointer into [ set<MapPoint*> mspMapPoints ]
					}
					size_t KPId; // KeyPoint Id
					ss >> KPId;
					MP->AddObservation(KF,KPId); // FIXME: this doesn't seem right !!! which MapPoint ?
					//KF->AddMapPoint(MP,KPId); // FIXME: can not add MapPoint to KF because it runs mvpMapPoints[idx]=pMP
					// where mvpMapPoints is private and its size is specified when constructing Frame using N = mvKeys.size();
					float u, v; // KeyPoint coordinates (FIXME: not used right now)
					ss >> u;
					ss >> v;
				}
			}
			//delete(MP);
		}

		fBundler.close();
		fCoords.close();

		return;
}
