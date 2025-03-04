
/*
 *
 * Utility functions to preprocess data before running the demo
 *
 * This includes:
 * 1- Handeling ORBSLAM2 Map
 * 2- Taking a lidar scan to a depth image
 * 3- Setting up Lidar parameters.
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-09-00
 *
 */

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <assert.h>

#include "Thirdparty/ORB_SLAM2/include/KeyFrame.h"
#include "Thirdparty/ORB_SLAM2/include/MapPoint.h"
#include "Thirdparty/ORB_SLAM2/include/Map.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>

#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>

#include "GroundRemoval.hpp"

typedef pcl::PointXYZI  PointType;
typedef ORB_SLAM2::Map Map;
typedef ORB_SLAM2::Frame Frame;
typedef ORB_SLAM2::KeyFrame KeyFrame;
typedef ORB_SLAM2::MapPoint MapPoint;
typedef ORB_SLAM2::KeyFrameDatabase KeyFrameDatabase;
typedef ORB_SLAM2::ORBVocabulary ORBVocabulary;

/// Image parameters
int WIDTH;
int HEIGHT;
float DEPTHFACTOR=1.0f;

/// HDL-64E parameters
extern const int N_SCAN = 64;
extern const int Horizon_SCAN = 4500;
extern const int groundScanInd = 25;
extern const float ang_res_x = 360.0/float(Horizon_SCAN);
extern const float ang_res_y = 26.9/float(N_SCAN-1);
extern const float ang_bottom = 15.0+0.1;
extern const float sensorMountAngle = 0.0;

/*
/// Velodyne to camera calibration : 2011_09_26
extern const cv::Mat R = (cv::Mat_<float>(3,3) << 7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02);
extern const cv::Mat t = (cv::Mat_<float>(3,1) << -4.069766e-03, -7.631618e-02, -2.717806e-01);
/// Camera to camera : 2011_09_26
extern const cv::Mat R_cam = (cv::Mat_<float>(3,3) << 0.9999, 0.0098, -0.0074, -0.0099, 0.9999, -0.0043, 0.0074, 0.0044, 1.0000);
/// Rectification transformation : 2011_09_26
extern const cv::Mat P_rect = (cv::Mat_<float>(3,4) << 721.5377,0,609.5593,0,0,721.5377,172.8540,0,0,0,1.0000,0);
*/

/// Velodyne to camera calibration : 2011_09_30
extern const cv::Mat R = (cv::Mat_<float>(3,3) << 7.027555e-03, -9.999753e-01, 2.599616e-05, -2.254837e-03, -4.184312e-05, -9.999975e-01, 9.999728e-01, 7.027479e-03, -2.255075e-03);
extern const cv::Mat t = (cv::Mat_<float>(3,1) << -7.137748e-03, -7.482656e-02, -3.336324e-01);
/// Camera to camera : 2011_09_30
extern const cv::Mat R_cam = (cv::Mat_<float>(3,3) << 9.999280e-01, 8.085985e-03, -8.866797e-03, -8.123205e-03, 9.999583e-01, -4.169750e-03, 8.832711e-03, 4.241477e-03, 9.999520e-01);
/// Rectification transformation : 2011_09_30
extern const cv::Mat P_rect = (cv::Mat_<float>(3,4) << 7.070912e+02, 0.000000e+00, 6.018873e+02, 0.000000e+00, 0.000000e+00, 7.070912e+02, 1.831104e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00);

/// Laser containers - PCL
PointType nanPoint; // fill in fullCloud at each iteration
pcl::PointCloud<PointType>::Ptr laserCloudIn; 
pcl::PointCloud<PointType>::Ptr groundCloud;  //ground point cloud
pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
pcl::PointCloud<PointType>::Ptr fullInfoCloud;  // same as fullCloud, but with intensity - range

/// Laser containers - OpenCV
cv::Mat rangeMat; // range matrix for range image
cv::Mat labelMat; // label matrix for segmentaiton marking
cv::Mat groundMat; // ground matrix for ground cloud marking

/// Functions
cv::Mat read(std::string im_name);
cv::Point project(cv::Mat point);    
void LoadORBSLAM(const std::string Foldername, std::vector<std::string>& vstrImageFilenamesRGB, 
				std::vector<std::string>& vstrImageFilenamesL, Map* pMap);
void LoadMap(const std::string &strBundlerFilename, 
				const std::string &strCoordsFilename, 
				Map* mpMap, KeyFrameDatabase *KFDB);
void UpdateCameraGraph(const std::string strGraphFilename, Map *pMap);
void LoadImageNames(const std::string &strFrameIdFilename, 
				std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesL);
void LoadCategories(unordered_map<int,std::string> &categories, 
				const std::string &strCategoriesFilename);

cv::Mat GetRangeImageFromBinaryFile(const std::string vstrImageFilenamesL);
void allocateMemory();
void reset();
void projectPointCloud();
void groundRemoval();
