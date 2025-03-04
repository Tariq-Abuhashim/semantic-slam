
#include <pangolin/pangolin.h>
#include <thread>

#include "DoN.hpp"
#include "CloudViewer.hpp"

typedef pcl::PointXYZI PointType;

pcl::PointCloud<PointType>::Ptr laserCloudIn;
float scale1 = 0.1f;
float scale2 = 3.0f;
float donthres = 0.25f;
float segradius = 0.2f;

size_t imwidth = 1241;
size_t imheight = 376;

/*
*	Get lidar point cloud
*/
void GetCloudFromBinaryFile(const std::string Filename)
{
	std::fstream file(Filename, std::ios::in | std::ios::binary);
	if(file.good())
	{
		file.seekg(0, std::ios::beg);
		int i;
		for (i = 0; file.good() && !file.eof(); i++) 
		{
			PointType thisPoint;
			file.read((char *) &thisPoint.x, 3*sizeof(float));
			file.read((char *) &thisPoint.intensity, sizeof(float)); // only with pcl::PointXYZI
			laserCloudIn->push_back(thisPoint);
		}
		file.close();
	}
	else
	{
		std::cerr << std::endl << "Failed to open Binary file " << Filename << std::endl << std::endl;
	}
}

cv::Point project(cv::Mat point3d)
{
	/// Velodyne to camera calibration
	//const cv::Mat R = (cv::Mat_<float>(3,3) << 7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 
	//							7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02);
	//const cv::Mat t = (cv::Mat_<float>(3,1) << -4.069766e-03, -7.631618e-02, -2.717806e-01);
	const cv::Mat R = (cv::Mat_<float>(3,3) <<  0.0000, -1.0000,  0.0000,  
                                                0.0000,  0.0000, -1.0000, 
                                                1.0000,  0.0000,  0.0000);
	const cv::Mat t = (cv::Mat_<float>(3,1) << 0.0, 0.0, 0.0);
	/// Camera to camera
	//const cv::Mat R_cam = (cv::Mat_<float>(3,3) << 0.9999, 0.0098, -0.0074, -0.0099, 0.9999, -0.0043, 0.0074, 
	//							0.0044, 1.0000);
	const cv::Mat R_cam = (cv::Mat_<float>(3,3) << 1.0000, 0.0000, 0.0000, 
                                                   0.0000, 1.0000, 0.0000, 
                                                   0.0000, 0.0000, 1.0000);
	/// Rectification transformation
	//const cv::Mat P_rect = (cv::Mat_<float>(3,4) << 721.5377,0,609.5593,0,0,721.5377,172.8540,0,0,0,1.0000,0);
	const cv::Mat P_rect = (cv::Mat_<float>(3,4) << 690,0,imwidth/2,0,0,690,imheight/2,0,0,0,1.0000,0);

	cv::Mat Tr_velo_to_cam = cv::Mat::eye(4,4,CV_32F);
	R.copyTo(Tr_velo_to_cam.rowRange(0,3).colRange(0,3));
	t.copyTo(Tr_velo_to_cam.rowRange(0,3).col(3));

	cv::Mat T_cam = cv::Mat::eye(4,4,CV_32F);
	R_cam.copyTo(T_cam.rowRange(0,3).colRange(0,3));

	cv::Mat P_velo_to_img = P_rect*T_cam*Tr_velo_to_cam;

	cv::Mat point2d = P_velo_to_img*point3d;

	if (point2d.at<float>(0)<2)
		return cv::Point(-1,-1);

    return cv::Point(point2d.at<float>(0)/point2d.at<float>(2),point2d.at<float>(1)/point2d.at<float>(2));
}

int main ( int argc, char** argv )
{

	if( argc != 2) 
	{
		std::cout <<" Usage: ./testDoN /path/to/cloud.bin" << std::endl;
		std::cout <<" Example: ./testDoN /home/mrt/Data/kitti/2011_09_26/2011_09_26_drive_0035_sync/velodyne_points/data/0000000000.bin" << std::endl;
		return -1;
    }

	// allocate memory
	laserCloudIn.reset(new pcl::PointCloud<PointType>());

	// read scan file
	GetCloudFromBinaryFile(std::string(argv[1]));

	// initialise don
	DoN* don = new DoN(scale1, scale2, donthres, segradius);

	// crop data to image field of view (for presentation slides only)
	
	// get depth image and crop cloud to image FOV
	pcl::PointCloud<PointType>::Ptr NewCloud;
	NewCloud.reset(new pcl::PointCloud<PointType>());
	cv::Mat image = cv::Mat::zeros(imheight,imwidth,CV_32FC3);
	for (pcl::PointCloud<PointType>::iterator ptr = laserCloudIn->begin(); ptr!=laserCloudIn->end(); ptr++)
	{
		cv::Mat point3d = (cv::Mat_<float>(4,1) << (*ptr).x, (*ptr).y, (*ptr).z, 1);
		cv::Point point2d(project(point3d));
		if( point2d.x>0 && point2d.x<imwidth && point2d.y>0 && point2d.y<imheight )
		{
			float range = sqrt((*ptr).x*(*ptr).x + (*ptr).y*(*ptr).y + (*ptr).z*(*ptr).z);
			if (range>0)
			{
				PointType thisPoint = (*ptr);
				NewCloud->push_back(thisPoint);
				int intensity = 255*range/50; // assuming 50 meters max range
				cv::Scalar color = cv::Scalar(intensity,intensity,intensity);
				cv::circle(image, point2d, 1, color, -1, 2); // depth scaled for visualisation
				//image.at<cv::Vec3b>(point2d) = color; // FIXME: doesnt work
			}
		}
	}
	laserCloudIn = NewCloud;

	std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(1);
	cv::imwrite("lidar_depth_image.png", image, compression_params);
	
	// initialize the Viewer for pcl::PointXYZI
	//CloudViewer<PointType>* viewer = new CloudViewer<pcl::PointXYZI>(laserCloudIn);

	// perform segmentation
	don->extract(laserCloudIn);

	// initialize the Viewer for pcl::PointNormal
	CloudViewer<pcl::PointNormal>* viewer = new CloudViewer<pcl::PointNormal>(don->GetDonCloud(), don->GetClustersIndices());

	// initialize thread and launch
	std::thread* tviewer = new std::thread(&CloudViewer<pcl::PointNormal>::Run, viewer);
	
	while(1)
	{
	}

	return 0;
}
