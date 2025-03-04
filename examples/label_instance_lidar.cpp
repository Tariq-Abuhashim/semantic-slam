
/*
 * @file label_instance_lidar.cpp
 * Semantic SLAM using lidar example.
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-09-00
 */

#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>

#include "Engine.hpp"
#include "Utility.hpp"

// Settings file
std::string strSettingsFile = "../config/KITTI03.yaml";
// Categories file
std::string strCategoriesFilename = "../config/categories.txt";

int main ( int argc, char** argv ) 
{
	
	srand (static_cast <unsigned> (time(0)));

	if( argc != 2) 
	{
		std::cout << " Usage: ./label_instance_lidar /path/to/result_and_config/folder" << std::endl;
		std::cout << " Example: ./label_instance_lidar /home/mrt/Data/kitti/2011_09_26/2011_09_26_drive_0035_sync" 
				  << std::endl;
		return -1;
    }

	// Load ORB_SLAM2 Map (from stereo images)
	std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesL;
	ORB_SLAM2::Map* pMap = new ORB_SLAM2::Map();
	LoadORBSLAM(std::string(argv[1]), vstrImageFilenamesRGB, vstrImageFilenamesL, pMap);

	// Load object categories
	std::unordered_map<int,std::string> categories;
	LoadCategories(categories, strCategoriesFilename);

	// Summary
	std::vector<ORB_SLAM2::KeyFrame*> AllKeyFrames;
	AllKeyFrames = pMap->GetAllKeyFrames();
	std::cout << endl << "-------" << endl;
	std::cout << " * KeyFrames in the sequence: " << AllKeyFrames.size() << endl;
    std::cout << " * RGB images in the sequence: " << vstrImageFilenamesRGB.size() << endl;
	std::cout << " * D images in the sequence: " << vstrImageFilenamesL.size() << endl;
	std::cout << " * Labelled categories: " << categories.size() << endl;

    // Switch on the engine
	std::cout << endl << "-------" << endl;
	Engine engine(categories, strSettingsFile); // Create segmentation objects and visualisation threads
	std::cout << endl << "-------" << endl;

	// Allocate point cloud memory
	allocateMemory();

	// Navigate
	sort(AllKeyFrames.begin(), AllKeyFrames.end(), KeyFrame::lId); // sort KeyFrames according to Id
	for(size_t i=0; i<AllKeyFrames.size(); i++)
	//for(size_t i=0; i<900; i++)
	{
		KeyFrame *KF = AllKeyFrames[i]; // Get KeyFrame
		
		cv::Mat imRGB; // Get RGB image
        imRGB = read(std::string(argv[1])+vstrImageFilenamesRGB[i]);

		WIDTH = imRGB.cols;
		HEIGHT = imRGB.rows;

		cv::Mat imD; // Get depth image (updates: laserCloudIn)
        imD = GetRangeImageFromBinaryFile(std::string(argv[1])+vstrImageFilenamesL[i]);

/*
		GroundRemoval* SegGrnd = new GroundRemoval();
		std::vector<pcl::PointCloud<PointType>::Ptr> cloud_clusters;
		SegGrnd->segment(*laserCloudIn, cloud_clusters);
*/

/*	
		// (updates: fullCloud, fullInfoCloud, rangeMat)
		projectPointCloud();

		// remove ground
		groundRemoval();
*/

/*
		// Get Depth image
		cv::Mat imD2 = cv::Mat::zeros(HEIGHT,WIDTH,CV_32F);
		for (pcl::PointCloud<PointType>::iterator ptr = cloud_clusters[1]->begin(); ptr!=cloud_clusters[1]->end(); ptr++)
		{
			cv::Mat point3d = (cv::Mat_<float>(4,1) << (*ptr).x, (*ptr).y, (*ptr).z, 1);
			cv::Point point2d(project(point3d));
			if( point2d.x>0 && point2d.x<WIDTH && point2d.y>0 && point2d.y<HEIGHT )
			{
				float depth = sqrt((*ptr).x*(*ptr).x + (*ptr).y*(*ptr).y + (*ptr).z*(*ptr).z);
				if (depth>0)
				{
					//std::cout << depth << " ";
					imD2.at<float>(round(point2d.y),round(point2d.x)) = depth;
				}
			}
		}
*/

		// label the ground points (updates: groundMat, labelMat, groundCloud)
		//std::vector<int> indices;
    	//pcl::removeNaNFromPointCloud(*fullCloud, *fullCloud, indices);

		//cv::namedWindow("imD");
		//cv::imshow("imD",imD);

		//cv::namedWindow("Ground");
		//cv::imshow("Ground",imD2);

        //cv::waitKey(0);


		std::cout << "Processing frame " << i << ":" << std::endl;
		engine.Run(imRGB, imD, KF); // Run the R-CNN segmentation Engine

		// Reset (Necessary to have)
		reset();
	}
	
	
	std::cout << std::endl;
	std::cout << "Demo finished." << std::endl;
	std::cout << "Press ctrl+c to exit ...." << std::endl;
	std::cout << std::endl;

	while(1) {};

	cv::waitKey(0);

	return 0;

}


