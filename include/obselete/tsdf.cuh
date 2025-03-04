/*
*
* Tariq Abuhashim
* t.abuhashim@gmail.com
* July, 2019
*
* Adapted from Andy Zeng, Princeton University, 2016
*
*/

#ifndef tsdf_CUH
#define tsdf_CUH

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel function to integrate a TSDF voxel volume given depth images
class TSDF
{
	public:

	TSDF(int h, int w);
	~TSDF();

	void Integrate(float *depth_im);
	float * voxel_grid_TSDF;
	float * voxel_grid_weight;

	private:

	float * gpu_voxel_grid_TSDF;
	float * gpu_voxel_grid_weight;
	std::vector<float> cam_K_vec;
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
  	float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
  	float voxel_grid_origin_y = -1.5f;
  	float voxel_grid_origin_z = 0.5f;
  	float voxel_size = 0.006f;
  	float trunc_margin = voxel_size * 5;
  	int voxel_grid_dim_x = 100;
  	int voxel_grid_dim_y = 100;
  	int voxel_grid_dim_z = 100;
	float * gpu_cam_K;
	float * gpu_cam2base;
	float * gpu_depth_im;
};

#endif // tsdf_CUH
