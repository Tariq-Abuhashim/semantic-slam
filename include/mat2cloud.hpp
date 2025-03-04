

#include <pcl/io/pcd_io.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // imread

#include <boost/filesystem.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

typedef struct Intr
{
    int width;
    int height;
    float fx;
    float fy;
    float cx;
    float cy;
    float scale_factor;
} Intr;

/*  Predefined data for freiburg dataset 3 
 *  */
/*
const Intr DEFAULT_CAM_PARAMS   =   {

                                    640,
                                    480,
                                    535.4f,
                                    539.2f,
                                    320.1f,
                                    247.6f,
                                    ( 1.f / 5000.f )
                                };
*/

/*  Predefined data for KITTI dataset 3 
 *  */
const Intr DEFAULT_CAM_PARAMS   =   {

                                    1241,
                                    376,
                                    721.5377f,
                                    721.5377f,
                                    609.5593f,
                                    172.8540f,
                                    ( 1.f )
                                };

template <class T>
void set_pixel(T & pcl_pixel, cv::Mat & src, int x, int y, Intr& cam_params)
{
	std::cerr << "set_pixel: Error - do not have proper specification for type: " << typeid(T).name() << std::endl;
	throw;
};

template <>
void set_pixel(pcl::RGB & pcl_color_pixel, cv::Mat & src, int x, int y, Intr& cam_params)
{
	uint32_t rgb;
	cv::Vec3b cur_rgb = src.at<cv::Vec3b>(y,x); // b,g,r
	rgb = (static_cast<int>(cur_rgb[ 2 ])) << 16 |
			(static_cast<int>(cur_rgb[ 1 ])) << 8 |
			(static_cast<int>(cur_rgb[ 0 ]));
	pcl_color_pixel.rgb = static_cast<uint32_t>(rgb);
};

template <>
void set_pixel(pcl::PointXYZ & xyz_pcl_pixel, cv::Mat & src, int x, int y, Intr& cam_params)
{
	xyz_pcl_pixel.z = src.at<unsigned short>( y * cam_params.width + x ) * cam_params.scale_factor;
	xyz_pcl_pixel.x = xyz_pcl_pixel.z * ( x - cam_params.cx ) / cam_params.fx;
	xyz_pcl_pixel.y = xyz_pcl_pixel.z * ( y - cam_params.cy ) / cam_params.fy;
};

template <class T>
bool load_cloud(cv::Mat Image, pcl::PointCloud<T>& pcl_cloud , Intr& cam_params)
{
	cv::Size im = Image.size();
	int width = im.width;
	int height = im.height;
	int nchannels = Image.channels();
	int step = Image.step;

	pcl_cloud.width = width;
	pcl_cloud.height = height;
	pcl_cloud.is_dense = true;
	pcl_cloud.points.resize(width*height);

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			T current_pixel;
			set_pixel<T>(current_pixel, Image, x, y, cam_params );
			pcl_cloud(x,y) = current_pixel;
        }
};
