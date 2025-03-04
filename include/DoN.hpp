#ifndef DON_H
#define DON_H

#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/don.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/opencv.hpp>

#include <map>


class DoN 
{

	private:

	/// The smallest scale to use in the DoN filter.
	double mnscale1;

	/// The largest scale to use in the DoN filter.
	double mnscale2;

	/// The minimum DoN magnitude to threshold by.
	double mnthreshold;

	/// Segment scene into clusters with given distance tolerance using euclidean clustering.
	double mnsegradius;

	/// DoN map computed using DoN operator (equation 2).
	pcl::PointCloud<pcl::PointNormal>::Ptr mpdoncloud;

	/// Segmentation of the DoN map based on threshold over DoN magnitude.
	std::vector<pcl::PointIndices> mvcluster_indices;

	/// Input image width and height.
	int mnwidth, mnheight;

	/// Camera parameters, the scale is to down-sample the image?
	float mnfx, mnfy, mncx, mncy, mnscale;

	/// Minimum and maximum acceptable cluster size.
	int mnMinClusterSize, mnMaxClusterSize;

	/// Sensor type
	int mSensor;

	/// A boolean flag to show debugging messages.
	bool verbose;
	
	public:

	/// Default constructor.
	DoN();

	/// Constructor with inputs.
	DoN(double scale1, double scale2, double threshold, double segradius);

	/// Constructor with more image inputs.
	DoN(double scale1, double scale2, double threshold, double segradius, cv::Mat K, int imWidth, int imHeight, int);

	/// Default destructor.
	~DoN();

	/// The main function to cluster the depth data.
	std::map<int, std::vector<cv::Point> > extract(cv::Mat& imRGB, cv::Mat& imD);

	void extract(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
	
	/// Obselete function: returns a vector of clusters in image coordinates.
	std::map<int, std::vector<cv::Point> > GetClusters();

	/// 2D visualisation function, which requires and input image to display clusters on.
	void show2d(cv::Mat& imRGB);

	/// 3D visualisation function.
	void show3d();

	pcl::PointCloud<pcl::PointNormal>::Ptr GetDonCloud();
	std::vector<pcl::PointIndices> GetClustersIndices();

	private:

	/// Function that turns a pair of input RGB and D images into a pcl::PointXYZRGB point-cloud.
	void loadCloud(cv::Mat& imRGB, cv::Mat& imD, pcl::PointCloud<pcl::PointXYZRGB>& cloud);

};


#endif //DON_H
