/*
 * Copyright (C) 2019 by AutoSense Organization. All rights reserved.
 * Gary Chan <chenshj35@mail2.sysu.edu.cn>
 */
#ifndef SEGMENTERS_INCLUDE_SEGMENTERS_GROUND_PLANE_FITTING_SEGMENTER_HPP_
#define SEGMENTERS_INCLUDE_SEGMENTERS_GROUND_PLANE_FITTING_SEGMENTER_HPP_

#include <Eigen/Core>
#include <string>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/extract_indices.h>  // pcl::ExtractIndices
#include <pcl/io/io.h>                    // pcl::copyPointCloud

typedef pcl::PointXYZI  PointType;

template <typename PointType>
	bool sortByAxisZAsc(PointType p1, PointType p2) {
    	return p1.z < p2.z;
	}

	template <typename PointType>
	bool sortByAxisXAsc(PointType p1, PointType p2) {
    	return p1.x < p2.x;
	}

typedef struct {
    Eigen::MatrixXf normal;
    double d = 0.;
} model_t;

/**
 * @brief Ground Removal based on Ground Plane Fitting(GPF)
 * @refer
 *   Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for
 * Autonomous Vehicle Applications (ICRA, 2017)
 */
class GroundRemoval {
 public:
    GroundRemoval();
    ~GroundRemoval();

    /// @brief Segment the point cloud.
    void segment(
        const pcl::PointCloud<PointType> &cloud_in,
        std::vector<pcl::PointCloud<PointType>::Ptr> &cloud_clusters);

 private:
    void extractInitialSeeds(const pcl::PointCloud<PointType> &cloud_in,
                             pcl::PointCloud<PointType>::Ptr cloud_seeds);

    model_t estimatePlane(const pcl::PointCloud<PointType> &cloud_ground);

    void mainLoop(const pcl::PointCloud<PointType> &cloud_in,
                  pcl::PointCloud<PointType>::Ptr cloud_gnds,
                  pcl::PointCloud<PointType>::Ptr cloud_ngnds);

	// Ground Plane Fitting Segmenter
	// in Paper: Nsegs=3/Niter=3/Nlpr=20/THseeds=0.4m/THdist=0.2m
	 const float gpf_sensor_height = 1.73;
	 const int gpf_num_segment= 3;
	 const int gpf_num_iter=10 ; // number of iterations
	 const int gpf_num_lpr= 20; // number of points used to estimate the lowest point representative(LPR)
	 const float gpf_th_lprs= 0.08;
	 const float gpf_th_seeds= 0.4; // threshold for points to be considered initial seeds
	 const float gpf_th_gnds= 0.1; // ground points threshold distance from the plane <== large to guarantee safe removal

 private:
    //static const int segment_upper_bound_ = 6;
};  // class GroundRemoval

#endif  // SEGMENTERS_INCLUDE_SEGMENTERS_GROUND_PLANE_FITTING_SEGMENTER_HPP_
