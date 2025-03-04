
#include "Utility.hpp"

/*	
	// Count the number of data files in the folder
	int _number_of_frames = 0;
    boost::filesystem::directory_iterator dit(string(argv[1])+"/velodyne_points/data/");
    boost::filesystem::directory_iterator eit;
    while(dit != eit)
    {
        if(boost::filesystem::is_regular_file(*dit) && dit->path().extension() == ".bin")
        {
            _number_of_frames++;
        }
        ++dit;
    }
*/


// FIXME: REVIEW
/*
*	read RGB and Depth images
*/
cv::Mat read(std::string im_name)
{
	cv::Mat im;
	im = cv::imread(im_name, CV_LOAD_IMAGE_UNCHANGED);
    if(im.empty()) 
	{
		std::cerr << std::endl << "Failed to load image at: "
			<< im_name << std::endl;
		exit(-1);
    }
	return im;
}

/*
*
*/
cv::Point project(cv::Mat point3d)
{
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

// FIXME: REVIEW
/*
*	LoadORBSLAM
*/
void LoadORBSLAM(const std::string Foldername, std::vector<std::string>& vstrImageFilenamesRGB, 
				std::vector<std::string>& vstrImageFilenamesL, Map* pMap)
{
	std::cout << std::endl << "-------" << std::endl;
    std::cout << " * Loading ORB Vocabulary. This could take a while ... ";
	std::string strVocFilename = "../config/ORBvoc.txt";
    ORBVocabulary* mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFilename);
    if(!bVocLoad) {
        std::cerr << std::endl << "Wrong path to vocabulary. " << std::endl;
        std::cerr << "Falied to open at: " << strVocFilename << std::endl;
        exit(-1);}

	KeyFrameDatabase *KFDB = new KeyFrameDatabase(*mpVocabulary);
    std::cout << "Done !" << endl;

	std::cout << " * Loading Map from bundler file ... ";
	std::string strBundleFilename = Foldername+"/slam/bundle.txt";
	std::string strCoordsFilename = Foldername+"/slam/coords.txt";
	LoadMap(strBundleFilename, strCoordsFilename, pMap, KFDB);
	std::cout << "Done !" << std::endl;

	std::cout << " * Updating Connected Componants from Camera Graph file file ... ";
	std::string strGraphFilename =  Foldername+"/slam/camera_graph.txt";
	UpdateCameraGraph(strGraphFilename, pMap);
	std::cout << "Done !" << std::endl;

	// Retrieve paths to KeyFrame images
	std::cout << " * Loading image paths ... ";
	std::string strFrameIdFilename = Foldername+"/slam/frame_id.txt";
    LoadImageNames(strFrameIdFilename, vstrImageFilenamesRGB, vstrImageFilenamesL);
    if(vstrImageFilenamesRGB.empty()) {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        exit(-1);}
    else if(vstrImageFilenamesL.size()!=vstrImageFilenamesRGB.size()) {
        std::cerr << std::endl << "Different number of images for rgb and lidar." << std::endl;
        exit(-1);}
	std::cout << "Done !" << std::endl;
}

/*
*	LoadMap
*/
void LoadMap(const std::string &strBundlerFilename, 
				const std::string &strCoordsFilename, Map* mpMap, KeyFrameDatabase *KFDB) 
{ 
	
	std::ifstream fBundler;
    fBundler.open(strBundlerFilename.c_str());
	if(!fBundler.is_open())
	{
		std::cerr << std::endl << "Failed to load bundler file at: " << strBundlerFilename << std::endl;
		return;
	}
	std::ifstream fCoords;
	fCoords.open (strCoordsFilename.c_str());
	if (!fCoords.is_open())
	{
		std::cerr << std::endl << "Failed to load coords file at: " << strBundlerFilename << std::endl;
		return;
	}

	std::string s;
		
	size_t m, n; // sizes (KeyFrames, MapPoints)
	std::getline(fBundler,s);
	if(!s.empty())
	{
		std::stringstream ss;
		ss << s;
		ss >> m;
		ss >> n;
	}

	for(size_t i=0; i<m; i++)
	{
		cv::Mat Rcw(3,3,CV_32F);//
		cv::Mat tcw(3,1,CV_32F);//

		std::getline(fBundler,s); // Intrinsics

		for(int j=0; j<3; j++) { // Three lines 
			std::getline(fBundler,s); // Rotation - rows
			if(!s.empty()) {
				std::stringstream ss;
				for(int k=0; k<3; k++) {
					ss << s;  
					ss >> Rcw.at<float>(j,k);
				}
			}
		}

		std::getline(fBundler,s); // One line
		if(!s.empty()) {  // Translation
			std::stringstream ss;
			for(int j=0; j<3; j++) {
				ss << s;  
				ss >> tcw.at<float>(j);
			}
		}

		cv::Mat Tcw;
		Tcw = cv::Mat::eye(4,4,CV_32F);
		Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
		tcw.copyTo(Tcw.rowRange(0,3).col(3));

		// Create a Frame
		Frame CurrentFrame;
		CurrentFrame.mnId = i; // This affects KF->mnFrameId
		CurrentFrame.SetPose(Tcw);

		std::getline(fCoords, s);
		size_t camId;
		size_t index = s.find("#index = "); // first occurance
		if (index != std::string::npos) 
		{
			size_t comma = s.find(",", index+1);
			if (comma != std::string::npos) 
			{
				std::string ncam(s,index+9,comma-(index+9));
				std::stringstream ss(ncam);
				ss >> camId;
			}
		}
	
		assert(camId!=i); // FIXME: Why not working ?

		size_t npts;
		size_t keys = s.find("keys = "); // first occurance
		if (keys != std::string::npos) 
		{
			size_t comma = s.find(",", keys+1);
			if (comma != std::string::npos) 
			{
				std::string npts_str(s,keys+7,comma-(keys+7));
				std::stringstream ss(npts_str);
				ss >> npts;
			}
		}

		for(size_t i=0; i<npts; i++)
		{
			std::getline(fCoords, s);
			std::stringstream ss;
			ss << s;
			size_t KPId;
			ss >> KPId;
			float x, y;
			ss >> x;
			ss >> y;
			cv::KeyPoint *KP = new cv::KeyPoint();
			KP->pt.x = x;
			KP->pt.y = y;
			CurrentFrame.mvKeys.push_back(*KP);
			CurrentFrame.mvuRight.push_back(-10); // no right image - Monocular similar
			//cout << KPId << " " << x << " " << y << endl;
		}
    	
		// Create a KeyFrame
		KeyFrame* KF = new KeyFrame(CurrentFrame, mpMap, KFDB); // This affects KF->mnId
		//cout << "KeyFrame " << KF->mnId << " has " << KF->mvKeys.size() << " measurements." << endl;

		// Insert the KeyFrame in the map
		mpMap->AddKeyFrame(KF);  // Inserts the pointer into [ set<KeyFrame*> mspKeyFrames ]
		mpMap->mvpKeyFrameOrigins.push_back(KF);

		if (i==0) { }// check Tracking::StereoInitialization() to know what ORB_SLAM2::Map needs for initialisation
	}

	fBundler.close();
	fCoords.close();

	return;
}

void UpdateCameraGraph(const std::string strGraphFilename, Map *pMap)
{
	std::vector<KeyFrame*> vpKF = pMap->GetAllKeyFrames();
	std::sort(vpKF.begin(), vpKF.end(), KeyFrame::lId); // sort KeyFrames according to Id

	std::ifstream fGraph;
    fGraph.open(strGraphFilename.c_str());
	if(!fGraph.is_open())
	{
		std::cerr << std::endl << "Failed to load Graph file at: " << strGraphFilename << std::endl;
		return;
	}

	std::string s;
	for (int i=0; i<vpKF.size(); i++)
	{
		std::getline(fGraph, s);
		std::stringstream ss;
		ss << s;
		size_t parent;
		ss >> parent;
	
		KeyFrame* pKF = vpKF[parent];
		if (pKF->mnId != parent)
		{
			std::cerr << std::endl << "UpdateCameraGraph : Conflicting frames Id." << std::endl;
			exit(-1);
		}

		size_t nChildren;
		ss >> nChildren;

		std::cout << parent << " " << nChildren << " ";

		for (int j=0; j<nChildren; j++)
		{
			size_t child;
			ss >> child;
			int weight;
			ss >> weight;
			pKF->AddConnection(vpKF[child], weight);
			std::cout << child << " " << weight << " ";
		}
		
		std::cout << endl;

		//const vector<KeyFrame*> vCovKFs = pKF->GetCovisiblesByWeight(100);
		const set<KeyFrame*> sKFs = pKF->GetConnectedKeyFrames();
		std::cout << sKFs.size() << endl;
	}

}

// FIXME: REVIEW
/*
*	LoadImageNames
*/
void LoadImageNames(const string &strFrameIdFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesL) {
    std::ifstream f_name;
    f_name.open(strFrameIdFilename.c_str());
	if(!f_name.is_open())
	{
		std::cerr << std::endl << "Failed to load frame_id file at: " << strFrameIdFilename << std::endl;
		return;
	}
	std::string image_ext = ".png";
	std::string lidar_ext = ".bin";
    while(!f_name.eof()) {
		std::string image_data = "/image_02/data/";
		std::string lidar_data = "/velodyne_points/data/";
        std::string s;
        getline(f_name,s);
        if(!s.empty()) {
            std::stringstream ss;
			ss << s;
			int t;
			ss >> t;
            char FileId [11];
			sprintf(FileId, "%010d", t);
			image_data+=FileId;
			image_data+=image_ext;
			lidar_data+=FileId;
			lidar_data+=lidar_ext;
			//std::cout << image_data << std::endl;
			//std::cout << lidar_data << std::endl;
            vstrImageFilenamesRGB.push_back(image_data);
            vstrImageFilenamesL.push_back(lidar_data);
        }
    }
}

// FIXME: REVIEW
/*
*	LoadCategories
*/
void LoadCategories(std::unordered_map<int,std::string> &categories, 
					const std::string &strCategoriesFilename) {
	std::cout << " * Loading categories ... ";
	std::ifstream fCategories;
  	fCategories.open(strCategoriesFilename.c_str());
	if(fCategories.is_open())
	{
    	while(!fCategories.eof()) {
			std::string line;
			std::getline(fCategories,line);
			if(!line.empty()) {
            	std::stringstream ss;
            	ss << line;
            	std::string value;
            	int key;
            	ss >> value;
            	ss >> key;
				//std::cout << key << " " << value << std::endl;
            	categories.insert(make_pair<int,std::string>((int)key,(std::string)value));
        	}
		}
		fCategories.close();
	}
	else
	{
		std::cerr << std::endl << "Failed to load categories at: " << strCategoriesFilename << std::endl;
	}

	if(categories.empty()) {
		std::cerr << std::endl << "Empty categories at: " << strCategoriesFilename << std::endl;
		exit(-1);}
	std::cout << "Done !" << std::endl;

}


// FIXME: REVIEW
/*
*	Range image from lidar scan binary file name
*/
cv::Mat GetRangeImageFromBinaryFile(const std::string Filename)
{

	// Get lidar point cloud
	//pcl::PointCloud<PointType>::Ptr laserCloudIn(new pcl::PointCloud<PointType>);
	std::fstream file(Filename, std::ios::in | std::ios::binary);
	if(file.good())
	{
		file.seekg(0, std::ios::beg);
		int i;
		for (i = 0; file.good() && !file.eof(); i++) 
		{
			PointType thisPoint;
			file.read((char *) &thisPoint.x, 3*sizeof(float));
			file.read((char *) &thisPoint.intensity, sizeof(float));
			//std::cout << thisPoint.x << " " << thisPoint.y << " " << thisPoint.z << std::endl;
			laserCloudIn->push_back(thisPoint);
		}
		file.close();
	}
	else
	{
		std::cerr << std::endl << "Failed to open Binary file " << Filename << std::endl << std::endl;
	}

	// Get Depth image
	cv::Mat imD = cv::Mat::zeros(HEIGHT,WIDTH,CV_32F);
	for (pcl::PointCloud<PointType>::iterator ptr = laserCloudIn->begin(); ptr!=laserCloudIn->end(); ptr++)
	{
		cv::Mat point3d = (cv::Mat_<float>(4,1) << (*ptr).x, (*ptr).y, (*ptr).z, 1);
		cv::Point point2d(project(point3d));
		if( point2d.x>0 && point2d.x<WIDTH && point2d.y>0 && point2d.y<HEIGHT )
		{
			float range = sqrt((*ptr).x*(*ptr).x + (*ptr).y*(*ptr).y + (*ptr).z*(*ptr).z);
			if (range>0)
			{
				//std::cout << depth << " ";
				imD.at<float>(round(point2d.y),round(point2d.x)) = range;
			}
		}
	}
	//imD.convertTo(imD, CV_8U, DEPTHFACTOR);
			
	return imD;

}


void allocateMemory()
{

	laserCloudIn.reset(new pcl::PointCloud<PointType>());
	fullCloud.reset(new pcl::PointCloud<PointType>());
	fullInfoCloud.reset(new pcl::PointCloud<PointType>());
	groundCloud.reset(new pcl::PointCloud<PointType>());
        
	fullCloud->points.resize(N_SCAN*Horizon_SCAN);
	fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

	rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
	groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
	labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));

}

void reset()
{
	laserCloudIn->clear();
	groundCloud->clear();
	std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
	std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);

	rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
	groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
	labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
	//labelCount = 1;
}

void projectPointCloud()
{

	// range image projection
	float verticalAngle, horizonAngle, range;
	size_t rowIdn, columnIdn, index, cloudSize; 

	PointType thisPoint;

	cloudSize = laserCloudIn->points.size();

	for (size_t i = 0; i < cloudSize; ++i){

		thisPoint.x = laserCloudIn->points[i].x;
		thisPoint.y = laserCloudIn->points[i].y;
		thisPoint.z = laserCloudIn->points[i].z;
		// find the row and column index in the iamge for this point
		verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x*thisPoint.x + thisPoint.y*thisPoint.y))*180 / M_PI;
		rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
		if (rowIdn < 0 || rowIdn >= N_SCAN)
			continue;

		horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

		columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
		if (columnIdn >= Horizon_SCAN)
			columnIdn -= Horizon_SCAN;

		if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
			continue;

		range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
		if (range < 0.1)
			continue;
        
		rangeMat.at<float>(rowIdn, columnIdn) = range;

		thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

		index = columnIdn  + rowIdn * Horizon_SCAN;
		fullCloud->points[index] = thisPoint;
		fullInfoCloud->points[index] = thisPoint;
		fullInfoCloud->points[index].intensity = range;// corresponding range of a point is saved as "intensity"
	}
}

void groundRemoval()
{
	size_t lowerInd, upperInd;
	float diffX, diffY, diffZ, angle;

	//int labelCount;
	//labelCount = 1;

	// groundMat
	// -1, no valid info to check if ground of not
	//  0, initial value, after validation, means not ground
	//  1, ground
	for (size_t j = 0; j < Horizon_SCAN; ++j){
		for (size_t i = 0; i < N_SCAN; ++i){

			lowerInd = j + ( i )*Horizon_SCAN;
			upperInd = j + (i+1)*Horizon_SCAN;

			if (fullCloud->points[lowerInd].intensity == -1 ||
				fullCloud->points[upperInd].intensity == -1) {
					// no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
			}
                    
			diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
			diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
			diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

			angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

			if (abs(angle - sensorMountAngle) <= 2.5) {
				groundMat.at<int8_t>(i,j) = 1;
				//groundMat.at<int8_t>(i+1,j) = 1;
			}
		}
	}
	// extract ground cloud (groundMat == 1)
	// mark entry that doesn't need to label (ground and invalid point) for segmentation
	// note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
	for (size_t i = 0; i < N_SCAN; ++i){
		for (size_t j = 0; j < Horizon_SCAN; ++j){
			if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
				labelMat.at<int>(i,j) = -1;
			}
		}
	}
   
	for (size_t i = 0; i <= N_SCAN; ++i){
		for (size_t j = 0; j < Horizon_SCAN; ++j){
			if (groundMat.at<int8_t>(i,j) == 1)
				groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
		}
	}

}

