
#ifndef BASE_SEGMENTER_HPP_
#define BASE_SEGMENTER_HPP_



class BaseSegmenter {
 public:
    /// @brief Segment the point cloud.
    virtual void segment(
        const pcl::PointCloud<PointType> &cloud_in,
        std::vector<pcl::PointCloud<PointType>::Ptr> &cloud_clusters) = 0;

    virtual std::string name() const = 0;
};  // BaseSegmenter

#endif
