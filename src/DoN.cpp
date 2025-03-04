
/*
 * @file DoN.cpp
 * This is part of Semantic SLAM.
 * Difference of Normals Segmentation.
 *
 * @author Yani Ioannou/Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-11-10
 */

#include <string>
#include "DoN.hpp"


/*
* Default constructor (set up for Lidar)
*/
DoN::DoN() 
	: mnscale1(0.2), mnscale2(2.0), mnthreshold(0.25), mnsegradius(0.2),
	mnwidth(1241), mnheight(376), mnfx(721.5377f), mnfy(721.5377f), mncx(609.5593f), mncy(172.8540f), // lidar
	//mnwidth(640), mnheight(480), mnfx(535.4f), mnfy(539.2f), mncx(320.1f), mncy(247.6f), // rgbd
	mnscale(1.0f), mnMinClusterSize(15), mnMaxClusterSize(10000), verbose(false), mSensor(1) // Lidar 1, RGBD 2
{
	pcl::PointCloud<pcl::PointNormal>::Ptr dummy (new pcl::PointCloud<pcl::PointNormal>);
	mpdoncloud=dummy;
}

/*
* constructor for Lidar (PointCloud input using DoN::extract(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud))
*/
DoN::DoN(double scale1, double scale2, double threshold, double segradius) 
	: mnscale1(scale1), mnscale2(scale2), mnthreshold(threshold), mnsegradius(segradius),
	mnwidth(1241), mnheight(376), mnfx(721.5377f), mnfy(721.5377f), mncx(609.5593f), mncy(172.8540f), // lidar
	//mnwidth(640), mnheight(480), mnfx(535.4f), mnfy(539.2f), mncx(320.1f), mncy(247.6f), // rgbd 
	mnscale(1.0f), mnMinClusterSize(15), mnMaxClusterSize(10000), verbose(true), mSensor(1)
{
	pcl::PointCloud<pcl::PointNormal>::Ptr dummy (new pcl::PointCloud<pcl::PointNormal>);
	mpdoncloud=dummy;
}

/*
* constructor for Lidar and RGBD (Depth image input using DoN::extract(cv::Mat& imRGB, cv::Mat& imD)
*/
DoN::DoN(double scale1, double scale2, double threshold, double segradius, cv::Mat K, 
				int imWidth, int imHeight, int Sensor) 
	: mnscale1(scale1), mnscale2(scale2), mnthreshold(threshold), mnsegradius(segradius), mnwidth(imWidth), 
	mnheight(imHeight), mSensor(Sensor), mnscale(1.0f), mnMinClusterSize(15), mnMaxClusterSize(1000000), verbose(true)
{
	
	pcl::PointCloud<pcl::PointNormal>::Ptr dummy (new pcl::PointCloud<pcl::PointNormal>);
	mpdoncloud=dummy;

	mnfx = K.at<float>(0,0);
	mnfy = K.at<float>(1,1);
	mncx = K.at<float>(0,2);
	mncy = K.at<float>(1,2);
	
}

/*
*
*/
DoN::~DoN()
{
}

/*
*
*/
void
DoN::loadCloud(cv::Mat& imRGB, cv::Mat& imD, pcl::PointCloud<pcl::PointXYZRGB> &current_xyzrgb_cloud)
{

	cv::Size im = imD.size();
	int width = im.width;
	int height = im.height;

	//std::vector<cv::Point2d> cvpoints;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			pcl::PointXYZRGB thisPoint;
			
			float range = imD.at<float>( y * width + x );
			if (range==0) continue;

			thisPoint.x = ( x - mncx ) / mnfx;
			thisPoint.y = ( y - mncy ) / mnfy;

			float d;
			if(mSensor==1)
			{
				float rim = sqrt(thisPoint.x*thisPoint.x + thisPoint.y*thisPoint.y + 1);
	 			d = range*mnscale/rim; // FIXME: Lidar
			}
			else if(mSensor==2)
				d = range*mnscale; // FIXME: RGBD

			thisPoint.x = d*thisPoint.x;
			thisPoint.y = d*thisPoint.y;
			thisPoint.z = d;
				
			//std::cout << thisPoint.x << " " << thisPoint.y << " " << thisPoint.z << std::endl;

			uint32_t rgb;
			cv::Vec3b cur_rgb = imRGB.at<cv::Vec3b>(y,x); // b,g,r
			rgb = (static_cast<int>(cur_rgb[ 2 ])) << 16 |
					(static_cast<int>(cur_rgb[ 1 ])) << 8 |
					(static_cast<int>(cur_rgb[ 0 ]));
			thisPoint.rgb = static_cast<uint32_t>(rgb);

			current_xyzrgb_cloud.push_back(thisPoint);

			//cvpoints.push_back(cv::Point(x,y));
        }
	}

	current_xyzrgb_cloud.width  = current_xyzrgb_cloud.points.size();
  	current_xyzrgb_cloud.height = 1;

}


/*
*
*/
std::map<int, std::vector<cv::Point> > 
DoN::extract(cv::Mat& imRGB, cv::Mat& imD)
{

	// Reset from previous runs
	mpdoncloud->clear();
	mvcluster_indices.clear();

	// Convert from cv::Mat to pcl::PointXYZRGB
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	loadCloud(imRGB, imD, *cloud);

	//pcl::PCDWriter writer;
	//writer.write<pcl::PointXYZRGB>("cloud.pcd", *cloud, false); 

	if(verbose) {
		std::cout << "Pointcloud contains: " << cloud->points.size() << " data points." << std::endl;
	}


	// Create a search tree, use KDTree for non-organized data.
	pcl::search::Search<pcl::PointXYZRGB>::Ptr tree;
	if (cloud->isOrganized()) {
		tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZRGB>());
	}
	else {
		tree.reset(new pcl::search::KdTree<pcl::PointXYZRGB>(false));
	}
	// Set the input pointcloud for the search tree
	tree->setInputCloud(cloud);


	if (mnscale1 >= mnscale2) {
		std::cerr << "Error: Large scale must be > small scale!" << std::endl;
		exit (EXIT_FAILURE);
	}

	// Compute normals using both small and large scales at each point
	pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::PointNormal> ne;
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);

	/**
	* NOTE: setting viewpoint is very important, so that we can ensure
	* normals are all pointed in the same direction!
	*/
	ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 
					std::numeric_limits<float>::max());

	// Calculate normals with the small scale
	if(verbose) {
		std::cout << "Calculating normals for scale ..." << mnscale1 << std::endl;
	}
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale(new pcl::PointCloud<pcl::PointNormal>);
	ne.setRadiusSearch(mnscale1);
	ne.compute(*normals_small_scale); // compute3DCentroid, computeCovarianceMatrix and flipNormalTowardsViewpoint

	// Calculate normals with the large scale
	if(verbose) {
		std::cout << "Calculating normals for scale ..." << mnscale2 << std::endl;
	}
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale(new pcl::PointCloud<pcl::PointNormal>);
	ne.setRadiusSearch(mnscale2);
	ne.compute(*normals_large_scale);

	// Create output cloud for DoN results
	if(verbose) {
		std::cout << "Calculating DoN ... " << std::endl;
	}
	//pcl::PointCloud<pcl::PointNormal>::Ptr mpdoncloud(new pcl::PointCloud<pcl::PointNormal>);
	pcl::copyPointCloud(*cloud, *mpdoncloud);

	// Create DoN operator
	pcl::DifferenceOfNormalsEstimation<pcl::PointXYZRGB, pcl::PointNormal, pcl::PointNormal> don;
	don.setInputCloud(cloud);
	don.setNormalScaleLarge(normals_large_scale);
	don.setNormalScaleSmall(normals_small_scale);

	if (!don.initCompute ()) {
		std::cerr << "Error: Could not initialize DoN feature operator" << std::endl;
		exit (EXIT_FAILURE);
	}

	// Compute DoN
	don.computeFeature(*mpdoncloud);

	// Save DoN features
	//pcl::PCDWriter writer;
	//writer.write<pcl::PointNormal>("don.pcd", *mpdoncloud, false); 

	// Filter by magnitude

	if(verbose) {
		std::cout << "Filtering out DoN mag <= " << mnthreshold << "..." << std::endl;
	}

	// Build the condition for filtering
	pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionOr<pcl::PointNormal>());
	range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
					new pcl::FieldComparison<pcl::PointNormal>("curvature", pcl::ComparisonOps::GT, mnthreshold)));

	// Build the filter
	pcl::ConditionalRemoval<pcl::PointNormal> condrem;
	condrem.setCondition(range_cond);
	condrem.setInputCloud(mpdoncloud);

	pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered(new pcl::PointCloud<pcl::PointNormal>);

	// Apply filter
	condrem.filter(*doncloud_filtered);
	mpdoncloud = doncloud_filtered;

	if(verbose) {
		std::cout << "Filtered Pointcloud: " << mpdoncloud->points.size() << " data points." << std::endl;
	}

	//writer.write<pcl::PointNormal>("don_filtered.pcd", *mpdoncloud, false); 

	// Filter by magnitude

	if(verbose) {
		std::cout << "Clustering using EuclideanClusterExtraction with tolerance <= " << mnsegradius << "..." << 
		std::endl;
	}

	pcl::search::KdTree<pcl::PointNormal>::Ptr segtree(new pcl::search::KdTree<pcl::PointNormal>);
	segtree->setInputCloud(mpdoncloud);

	//std::vector<pcl::PointIndices> mvcluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointNormal> ec;
	ec.setClusterTolerance(mnsegradius);
	ec.setMinClusterSize(mnMinClusterSize);
	ec.setMaxClusterSize(mnMaxClusterSize);
	ec.setSearchMethod(segtree);
	ec.setInputCloud(mpdoncloud);
	ec.extract(mvcluster_indices);


	// get the clusters
	return GetClusters();
	
}

/*
*	This function takes a pointer to a point cloud and visualises the stages of the segmentation
*/
void DoN::extract(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{

	// Reset from previous runs
	mpdoncloud->clear();
	mvcluster_indices.clear();

	//pcl::PCDWriter writer;
	//writer.write<pointType>("cloud.pcd", *cloud, false); 

	if(verbose) {
		std::cout << "Pointcloud contains: " << cloud->points.size() << " data points." << std::endl;
	}


	// Create a search tree, use KDTree for non-organized data.
	pcl::search::Search<pcl::PointXYZI>::Ptr tree;
	if (cloud->isOrganized()) {
		tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZI>());
	}
	else {
		tree.reset(new pcl::search::KdTree<pcl::PointXYZI>(false));
	}
	// Set the input pointcloud for the search tree
	tree->setInputCloud(cloud);


	if (mnscale1 >= mnscale2) {
		std::cerr << "Error: Large scale must be > small scale!" << std::endl;
		exit (EXIT_FAILURE);
	}

	// Compute normals using both small and large scales at each point
	pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::PointNormal> ne;
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);

	/**
	* NOTE: setting viewpoint is very important, so that we can ensure
	* normals are all pointed in the same direction!
	*/
	ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 
					std::numeric_limits<float>::max());

	// Calculate normals with the small scale
	if(verbose) {
		std::cout << "Calculating normals for scale ..." << mnscale1 << std::endl;
	}
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale(new pcl::PointCloud<pcl::PointNormal>);
	ne.setRadiusSearch(mnscale1);
	ne.compute(*normals_small_scale); // compute3DCentroid, computeCovarianceMatrix and flipNormalTowardsViewpoint

	// Calculate normals with the large scale
	if(verbose) {
		std::cout << "Calculating normals for scale ..." << mnscale2 << std::endl;
	}
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale(new pcl::PointCloud<pcl::PointNormal>);
	ne.setRadiusSearch(mnscale2);
	ne.compute(*normals_large_scale);

	// Create output cloud for DoN results
	if(verbose) {
		std::cout << "Calculating DoN ... " << std::endl;
	}
	pcl::copyPointCloud(*cloud, *mpdoncloud);

	// Create DoN operator
	pcl::DifferenceOfNormalsEstimation<pcl::PointXYZI, pcl::PointNormal, pcl::PointNormal> don;
	don.setInputCloud(cloud);
	don.setNormalScaleLarge(normals_large_scale);
	don.setNormalScaleSmall(normals_small_scale);

	if (!don.initCompute ()) {
		std::cerr << "Error: Could not initialize DoN feature operator" << std::endl;
		exit (EXIT_FAILURE);
	}

	// Compute DoN
	don.computeFeature(*mpdoncloud);

	//return;

	// Save DoN features
	//pcl::PCDWriter writer;
	//writer.write<pcl::PointNormal>("don.pcd", *mpdoncloud, false); 

	// Filter by magnitude

	if(verbose) {
		std::cout << "Filtering out DoN mag <= " << mnthreshold << "..." << std::endl;
	}

	// Build the condition for filtering
	pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionOr<pcl::PointNormal>());
	range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
					new pcl::FieldComparison<pcl::PointNormal>("curvature", pcl::ComparisonOps::GT, mnthreshold)));

	// Build the filter
	pcl::ConditionalRemoval<pcl::PointNormal> condrem;
	condrem.setCondition(range_cond);
	condrem.setInputCloud(mpdoncloud);

	pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered(new pcl::PointCloud<pcl::PointNormal>);

	// Apply filter
	condrem.filter(*doncloud_filtered);
	mpdoncloud = doncloud_filtered;

	if(verbose) {
		std::cout << "Filtered Pointcloud: " << mpdoncloud->points.size() << " data points." << std::endl;
	}

	//writer.write<pcl::PointNormal>("don_filtered.pcd", *mpdoncloud, false); 

	// Filter by magnitude

	if(verbose) {
		std::cout << "Clustering using EuclideanClusterExtraction with tolerance <= " << mnsegradius << "..." << 
		std::endl;
	}

	pcl::search::KdTree<pcl::PointNormal>::Ptr segtree(new pcl::search::KdTree<pcl::PointNormal>);
	segtree->setInputCloud(mpdoncloud);

	//std::vector<pcl::PointIndices> mvcluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointNormal> ec;
	ec.setClusterTolerance(mnsegradius);
	ec.setMinClusterSize(mnMinClusterSize);
	ec.setMaxClusterSize(mnMaxClusterSize);
	ec.setSearchMethod(segtree);
	ec.setInputCloud(mpdoncloud);
	ec.extract(mvcluster_indices);
	
}

/*
*
*/
std::map<int, std::vector<cv::Point> >
DoN::GetClusters()
{
	std::map<int, std::vector<cv::Point> > mClusters;
  	//std::unordered_set<std::vector<cv::Point> >::iterator it;

	// FIXME: a more efficient way is to save data indices while loading pointcloud from the image, and then return 		those instead.
	// Notice that some points may have been filtered out, so doncloud may not be in the same order and size as 		current_xyzrgb_cloud
	int index = 0;

	for(std::vector<pcl::PointIndices>::const_iterator it=mvcluster_indices.begin();it!=mvcluster_indices.end();++it)
	{
		std::vector<cv::Point> vCluster;
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			float x = mpdoncloud->points[*pit].x;
			float y = mpdoncloud->points[*pit].y;
			float z = mpdoncloud->points[*pit].z;

			int u = mnfx*x/z + mncx;
			int v = mnfy*y/z + mncy;

			vCluster.push_back(cv::Point(u,v));
		}
		mClusters.insert( std::pair<int, std::vector<cv::Point> >(index++, vCluster) );
	}

	return mClusters;
}

/*
*
*/
pcl::PointCloud<pcl::PointNormal>::Ptr DoN::GetDonCloud()
{
	return mpdoncloud;
}

/*
*
*/
std::vector<pcl::PointIndices> DoN::GetClustersIndices()
{
	return mvcluster_indices;
}

/*
*
*/
void
DoN::show2d(cv::Mat& imRGB)
{

	cv::Mat image = imRGB.clone();

	for(std::vector<pcl::PointIndices>::const_iterator it=mvcluster_indices.begin();it!=mvcluster_indices.end();++it)
	{
		cv::Scalar color = cv::Scalar(rand()%255, rand()%255, rand()%255);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			float x = mpdoncloud->points[*pit].x;
			float y = mpdoncloud->points[*pit].y;
			float z = mpdoncloud->points[*pit].z;

			int u = mnfx*x/z + mncx;
			int v = mnfy*y/z + mncy;

			cv::circle(image, cv::Point(u,v), 1, color, -1, 8);
		}
	}

	cv::namedWindow( "Geometric Segmentation", CV_WINDOW_AUTOSIZE );
	cv::imshow( "Geometric Segmentation", image );

	std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(1);
	cv::imwrite("DoN.png", image, compression_params);

	cv::waitKey(100);
	
}


/*
*
*/
void
DoN::show3d( )
{

	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  	viewer->setBackgroundColor(0,0,0);

	int j = 0;
	for(std::vector<pcl::PointIndices>::const_iterator it=mvcluster_indices.begin();it!=mvcluster_indices.end();++it,j++)
	{
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_cluster_don (new pcl::PointCloud<pcl::PointNormal>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
		  cloud_cluster_don->points.push_back (mpdoncloud->points[*pit]);
		}

		cloud_cluster_don->width = int (cloud_cluster_don->points.size());
		cloud_cluster_don->height = 1;
		cloud_cluster_don->is_dense = true;

		//Save cluster
		//std::cout<<"PointCloud representing the Cluster: "<<cloud_cluster_don->points.size()<<
		//			" data points."<<std::endl;
		//std::stringstream ss;
		//ss << "don_cluster_" << j << ".pcd";
		//writer.write<pcl::PointNormal> (ss.str (), *cloud_cluster_don, false);

		pcl::visualization::PointCloudColorHandlerRandom<pcl::PointNormal> single_color(cloud_cluster_don);
		viewer->addPointCloud<pcl::PointNormal> (cloud_cluster_don, single_color, std::to_string(j));
	}

	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  	viewer->addCoordinateSystem(1.0);
  	viewer->initCameraParameters();

    //This will get called once per visualization iteration
    //viewer.runOnVisualizationThread (viewerPsycho);
    while (!viewer->wasStopped())
    {
    //you can also do cool processing here
    //FIXME: Note that this is running in a separate thread from viewerPsycho
    //and you should guard against race conditions yourself...
    //user_data++;
		viewer->spinOnce(100);
    	//std::this_thread::sleep_for(100ms);
    }
 
}
