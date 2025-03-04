// ---------------------------------------------------------
// Author: Andy Zeng, Princeton University, 2016
// ---------------------------------------------------------

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/** @brief CUDA kernel function to integrate a TSDF voxel volume given depth images
	@author Tariq Abuhashim (modified from Andy Zeng, Princeton University, 2016)
	@date August 2019
*/
class TSDF
{
	public:

	/** Object TSDF constructor
	@param h depth image height
	@param w depth image width
	*/
	TSDF(int h, int w, int id, std::vector<float> base2world_vec, std::vector<float> origin);

	~TSDF();

	/** Incrementally, this function can be called to integrate a depth image into an object TSDF
	@param depth_im pointer to depth image
	*/
	void Integrate(float *depth_im, std::vector<float> cam2world_vec);

	/// pointer to a TSDF voxel grid
	float * voxel_grid_TSDF;

	/// pointer to a TSDF voxel grid weights
	float * voxel_grid_weight;

	private:

	float * gpu_voxel_grid_TSDF;
	float * gpu_voxel_grid_weight;
	//std::vector<float> cam_K_vec;
	std::vector<float> base2world_vec;
	float base2world_inv[16] = {0};
	float cam_K[3 * 3];
  	float base2world[4 * 4];
  	float cam2base[4 * 4];
  	float cam2world[4 * 4];
  	int im_width;
  	int im_height;
  	//float depth_im[im_height * im_width];  // FIXME: Initialised from input
  	// Voxel grid parameters (change these to change voxel grid resolution, etc.)
  	float voxel_grid_origin_x;// = 0.0f; // Location of voxel grid origin in base frame camera coordinates
  	float voxel_grid_origin_y;// = 0.0f;
  	float voxel_grid_origin_z;// = 3.0f;
  	float voxel_size = 0.004f;
  	float trunc_margin = voxel_size * 5;
  	int voxel_grid_dim_x = 200;
  	int voxel_grid_dim_y = 200;
  	int voxel_grid_dim_z = 200;
	float * gpu_cam_K;
	float * gpu_cam2base;
	float * gpu_depth_im;
	int mnId;

	/// Compute surface points from TSDF voxel grid and save points to point cloud file
	void SaveVoxelGrid2SurfacePointCloud(const std::string &file_name, int voxel_grid_dim_x, int voxel_grid_dim_y,
			int voxel_grid_dim_z, float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, 
			float voxel_grid_origin_z, float * voxel_grid_TSDF, float * voxel_grid_weight, float tsdf_thresh, 
			float weight_thresh);

	/// Load an M x N matrix from a text file (numbers delimited by spaces/tabs)
	/// Return the matrix as a float vector of the matrix in row-major order
	std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N);

	/// Read a depth image with size H x W and save the depth values (meters) into a float array (row-major order)
	/// The depth image file is assumed to be in 16-bit PNG format, depth in millimeters
	void ReadDepth(std::string filename, int H, int W, float * depth);

	/// 4x4 matrix multiplication (matrices are stored as float arrays in row-major order)
	void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]);

	/// 4x4 matrix inversion (matrices are stored as float arrays in row-major order)
	bool invert_matrix(const float m[16], float invOut[16]);
	void FatalError(const int lineNumber = 0);
	void checkCUDA(const int lineNumber, cudaError_t status);

	/// Camera intrinsic parameters
	std::vector<float> cam_K_vec = {535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1};

};
#endif // UTILS_H
