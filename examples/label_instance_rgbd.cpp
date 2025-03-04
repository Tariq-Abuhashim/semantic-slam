
/*
 * @file label_instance_rgbd.cpp
 * Semantic SLAM using rgbd example.
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date July, 2019
 */

#include "Engine.hpp"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <set>

#include <opencv2/opencv.hpp>

#include "Thirdparty/ORB_SLAM2/include/KeyFrame.h"
#include "Thirdparty/ORB_SLAM2/include/MapPoint.h"
#include "Thirdparty/ORB_SLAM2/include/Map.h"

#include <assert.h>

cv::Mat read(std::string im_name);

void LoadORBSLAM(std::vector<std::string>& vstrImageFilenamesRGB, std::vector<std::string>& vstrImageFilenamesD, Map* ORBMap);

void LoadCategories(unordered_map<int,std::string> &categories, const std::string &strCategoriesFilename);

void LoadImageNames(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                	std::vector<std::string> &vstrImageFilenamesD);

void LoadMap(const std::string &BundlerFilename, const std::string &strCoordsFilename, Map* mpMap, KeyFrameDatabase *KFDB);

void UpdateCameraGraph(const std::string strGraphFilename, Map *pMap);



int main( int argc, char** argv ) 
{

	if( argc != 2) 
	{
		std::cout <<" Usage: ./label_instance_rgbd /path/to/result_and_config/folder" << std::endl;
		std::cout <<" Example: ./label_instance_rgbd /home/mrt/Data/rgbd_dataset_freiburg3_long_office_household" << std::endl;
		return -1;
    }

	// Rebuild the ORB_SLAM2 pMap (FIXME: ORB_SLAM2 System should be placed inside SSLAM Engine)
	std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
	Map* pMap = new Map();
	LoadORBSLAM(vstrImageFilenamesRGB, vstrImageFilenamesD, pMap);

	// Load object categories
	unordered_map<int,std::string> categories;
	std::string strCategoriesFilename = "../config/categories.txt";
	LoadCategories(categories, strCategoriesFilename);

	// Summary
	std::vector<KeyFrame*> AllKeyFrames;
	AllKeyFrames = pMap->GetAllKeyFrames();
	std::cout << std::endl << "-------" << std::endl;
	std::cout << " * KeyFrames in the sequence: " << AllKeyFrames.size() << std::endl;
    std::cout << " * RGB images in the sequence: " << vstrImageFilenamesRGB.size() << std::endl;
	std::cout << " * D images in the sequence: " << vstrImageFilenamesD.size() << std::endl;
	std::cout << " * Labelled categories: " << categories.size() << std::endl;

    std::string strSettingsFile = "../config/TUM3.yaml";

	std::cout << std::endl << "-------" << std::endl;
	Engine engine(categories, strSettingsFile); // Initialise MaskRCNN
	std::cout << std::endl << "-------" << std::endl;

	sort(AllKeyFrames.begin(), AllKeyFrames.end(), KeyFrame::lId); // sort KeyFrames according to Id

	for(size_t i=0; i<AllKeyFrames.size(); i++)
	{
		KeyFrame *KF = AllKeyFrames[i]; // Get KeyFrame
		
		cv::Mat imRGB; // Get RGB image
        imRGB = read(std::string(argv[1]) + "/" + vstrImageFilenamesRGB[i]);

		cv::Mat imD; // Get depth image
        imD = read(std::string(argv[1]) + "/" + vstrImageFilenamesD[i]);

		cv::Mat imDepth = cv::Mat::zeros(imD.rows,imD.cols,imD.type());
		//cv::Mat imD2 = cv::Mat::zeros(HEIGHT,WIDTH,CV_32F);
		for(int col=0; col<imD.cols; col+=3)
		{
			for(int row=0; row<imD.rows; row+=4)
			{
				imDepth.at<float>(row, col) = imD.at<float>(row, col);
			}
		}

		float DepthMapFactor = 5000.0f;
    	imDepth.convertTo(imDepth, CV_32F, 1.0f/DepthMapFactor);
		std::cout << "  *Depth map scaled using depth factor : " << DepthMapFactor << std::endl;

		//cv::namedWindow("Ground");
		//cv::imshow("Ground",imDepth);

        //cv::waitKey(0);

		std::cout << "Processing frame " << i << ":" << std::endl;
		engine.Run(imRGB, imDepth, KF); // Run the R-CNN segmentation Engine
	}

	return 0;

}



// FIXME: REVIEW
// Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
// The depth image file is assumed to be in 16-bit PNG forcv::Mat, depth in millimeters
void ReadDepth(std::string filename, int H, int W, float * depth) {
	cv::Mat depth_mat = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
	if (depth_mat.empty()) 
	{
		std::cout << "Error: depth image file not read!" << std::endl;
    	cv::waitKey(0);
	}
	for (int r = 0; r < H; ++r)
	for (int c = 0; c < W; ++c) 
	{
		depth[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c)) / 1000.0f;
		if(depth[r * W + c] > 6.0f) // Only consider depth < 6m
        	depth[r * W + c] = 0;
    }
}

// FIXME: REVIEW
/*
*	read RGB and Depth images
*/
cv::Mat read(std::string im_name)
{
	cv::Mat im;
	im = cv::imread(im_name, CV_LOAD_IMAGE_UNCHANGED);
    if(im.empty()) 
	{
		std::cerr << std::endl << "Failed to load image at: "
			<< im_name << std::endl;
		exit(-1);
    }
	return im;
}

// FIXME: REVIEW
/*
*	LoadORBSLAM
*/
void LoadORBSLAM(std::vector<std::string>& vstrImageFilenamesRGB, std::vector<std::string>& vstrImageFilenamesD, Map* pMap)
{
	std::cout << std::endl << "-------" << std::endl;
    std::cout << " * Loading ORB Vocabulary. This could take a while ... ";
	std::string strVocFilename = "../config/ORBvoc.txt";
    ORBVocabulary* mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFilename);
    if(!bVocLoad) {
        std::cerr << std::endl << "Wrong path to vocabulary. " << std::endl;
        std::cerr << "Falied to open at: " << strVocFilename << std::endl;
        exit(-1);}

	KeyFrameDatabase *KFDB = new KeyFrameDatabase(*mpVocabulary);
    std::cout << "Done !" << std::endl;

	std::cout << " * Loading Map from bundler file ... ";
	std::string strBundleFilename = "../result/rgbd/bundle.txt";
	std::string strCoordsFilename =  "../result/rgbd/coords.txt";
	LoadMap(strBundleFilename, strCoordsFilename, pMap, KFDB);
	std::cout << "Done !" << std::endl;

	std::cout << " * Updating Connected Componants from Camera Graph file file ... ";
	std::string strGraphFilename =  "../result/rgbd/camera_graph.txt";
	UpdateCameraGraph(strGraphFilename, pMap);
	std::cout << "Done !" << std::endl;

	// Retrieve paths to KeyFrame images
	std::cout << " * Loading image paths ... ";
	std::string strAssociationFilename = "../result/rgbd/associations.txt";
    LoadImageNames(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD);
    if(vstrImageFilenamesRGB.empty()) {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        exit(-1);}
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size()) {
        std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
        exit(-1);}
	std::cout << "Done !" << std::endl;
}

// FIXME: REVIEW
/*
*	LoadCategories
*/
void LoadCategories(unordered_map<int,std::string> &categories, 
					const std::string &strCategoriesFilename) {
	std::cout << " * Loading categories ... ";
	std::ifstream fCategories;
  	fCategories.open(strCategoriesFilename.c_str());
	if(fCategories.is_open())
	{
    	while(!fCategories.eof()) {
			std::string line;
			std::getline(fCategories,line);
			if(!line.empty()) {
            	std::stringstream ss;
            	ss << line;
            	std::string value;
            	int key;
            	ss >> value;
            	ss >> key;
				//std::cout << key << " " << value << std::endl;
            	categories.insert(make_pair<int,std::string>((int)key,(std::string)value));
        	}
		}
		fCategories.close();
	}
	else
	{
		std::cerr << std::endl << "Failed to load categories at: " << strCategoriesFilename << std::endl;
	}

	if(categories.empty()) {
		std::cerr << std::endl << "Empty categories at: " << strCategoriesFilename << std::endl;
		exit(-1);}
	std::cout << "Done !" << std::endl;

}

// FIXME: REVIEW
/*
*	LoadImageNames
*/
void LoadImageNames(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD) {
    std::ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
	if(!fAssociation.is_open())
	{
		std::cerr << std::endl << "Failed to load associations file at: " << strAssociationFilename << std::endl;
		return;
	}
    while(!fAssociation.eof()) {
        std::string s;
        std::getline(fAssociation,s);
        if(!s.empty()) {
            std::stringstream ss;
			ss << s;
			float t;
			std::string sRGB, sD;
			ss >> t;
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
			ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}

// FIXME: REVIEW
/*
*	LoadMap
*/
void LoadMap(const std::string &strBundlerFilename, const std::string &strCoordsFilename, Map* mpMap, KeyFrameDatabase *KFDB) { 
	
	std::ifstream fBundler;
    fBundler.open(strBundlerFilename.c_str());
	if(!fBundler.is_open())
	{
		std::cerr << std::endl << "Failed to load bundler file at: " << strBundlerFilename << std::endl;
		return;
	}
	std::ifstream fCoords;
	fCoords.open (strCoordsFilename.c_str());
	if (!fCoords.is_open())
	{
		std::cerr << std::endl << "Failed to load coords file at: " << strBundlerFilename << std::endl;
		return;
	}

	std::string s;
		
	size_t m, n; // sizes (KeyFrames, MapPoints)
	std::getline(fBundler,s);
	if(!s.empty())
	{
		std::stringstream ss;
		ss << s;
		ss >> m;
		ss >> n;
	}

	for(size_t i=0; i<m; i++)
	{
		cv::Mat Rcw(3,3,CV_32F);//
		cv::Mat tcw(3,1,CV_32F);//

		std::getline(fBundler,s); // Intrinsics

		for(int j=0; j<3; j++) { // Three lines 
			std::getline(fBundler,s); // Rotation - rows
			if(!s.empty()) {
				std::stringstream ss;
				for(int k=0; k<3; k++) {
					ss << s;  
					ss >> Rcw.at<float>(j,k);
				}
			}
		}

		std::getline(fBundler,s); // One line
		if(!s.empty()) {  // Translation
			std::stringstream ss;
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

		std::getline(fCoords, s);
		size_t camId;
		size_t index = s.find("#index = "); // first occurance
		if (index != std::string::npos) 
		{
			size_t comma = s.find(",", index+1);
			if (comma != std::string::npos) 
			{
				std::string ncam(s,index+9,comma-(index+9));
				std::stringstream ss(ncam);
				ss >> camId;
			}
		}
	
		assert(camId!=i); // FIXME: Why not working ?

		size_t npts;
		size_t keys = s.find("keys = "); // first occurance
		if (keys != std::string::npos) 
		{
			size_t comma = s.find(",", keys+1);
			if (comma != std::string::npos) 
			{
				std::string npts_str(s,keys+7,comma-(keys+7));
				std::stringstream ss(npts_str);
				ss >> npts;
			}
		}

		for(size_t i=0; i<npts; i++)
		{
			std::getline(fCoords, s);
			std::stringstream ss;
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
			CurrentFrame.mvuRight.push_back(-10); // no right image - Monocular similar
			//std::cout << KPId << " " << x << " " << y << std::endl;
		}
    	
		// Create a KeyFrame
		KeyFrame* KF = new KeyFrame(CurrentFrame, mpMap, KFDB); // This affects KF->mnId
		//std::cout << "KeyFrame " << KF->mnId << " has " << KF->mvKeys.size() << " measurements." << std::endl;

		// Insert the KeyFrame in the map
		mpMap->AddKeyFrame(KF);  // Inserts the pointer into [ set<KeyFrame*> mspKeyFrames ]
		mpMap->mvpKeyFrameOrigins.push_back(KF);

		if (i==0) { }// check Tracking::StereoInitialization() to know what ORB_SLAM2::Map needs for initialisation
	}

	fBundler.close();
	fCoords.close();

	return;
}


void UpdateCameraGraph(const std::string strGraphFilename, Map *pMap)
{
	std::vector<KeyFrame*> vpKF = pMap->GetAllKeyFrames();
	sort(vpKF.begin(), vpKF.end(), KeyFrame::lId); // sort KeyFrames according to Id

	std::ifstream fGraph;
    fGraph.open(strGraphFilename.c_str());
	if(!fGraph.is_open())
	{
		std::cerr << std::endl << "Failed to load Graph file at: " << strGraphFilename << std::endl;
		return;
	}

	std::string s;
	for (int i=0; i<vpKF.size(); i++)
	{
		std::getline(fGraph, s);
		std::stringstream ss;
		ss << s;
		size_t parent;
		ss >> parent;
	
		KeyFrame* pKF = vpKF[parent];
		if (pKF->mnId != parent)
		{
			std::cerr << std::endl << "UpdateCameraGraph : Conflicting frames Id." << std::endl;
			exit(-1);
		}

		size_t nChildren;
		ss >> nChildren;

		std::cout << parent << " " << nChildren << " ";

		for (int j=0; j<nChildren; j++)
		{
			size_t child;
			ss >> child;
			int weight;
			ss >> weight;
			pKF->AddConnection(vpKF[child], weight);
			std::cout << child << " " << weight << " ";
		}
		
		std::cout << std::endl;

		//const std::vector<KeyFrame*> vCovKFs = pKF->GetCovisiblesByWeight(100);
		const set<KeyFrame*> sKFs = pKF->GetConnectedKeyFrames();
		std::cout << sKFs.size() << std::endl;
	}

}
