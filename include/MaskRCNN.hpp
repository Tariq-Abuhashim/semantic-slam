
/*
*
* Tariq Abuhashim
* t.abuhashim@gmail.com
* July, 2019
*
*/

#ifndef MaskRCNN_H
#define MaskRCNN_H

#include <stdio.h>
#include <Python.h>
#include <unicodeobject.h>
#include <opencv2/opencv.hpp>

#include <unordered_map>
#include <string>

#include <mutex>

#include "Thirdparty/ORB_SLAM2/include/KeyFrame.h"

/** @brief An interface between maskrcnn-benchmark in python and Engine in C++
	@author Tariq Abuhashim
	@date August 2019
*/
class MaskRCNN 
{

	public:
	/*
		    background = 0, 
			person = 1, bicycle = 2, car = 3, motorcycle = 4, airplane = 5, 
			bus = 6, train = 7, truck = 8, boat = 9, traffic light = 10,
		    fire hydrant = 11, stop sign = 12, parking meter = 13, bench = 14, bird = 15,
		    cat = 16, dog = 17, horse = 18, sheep = 19, cow = 20, 
			elephant = 21, bear = 22, zebra = 23, giraffe = 24, backpack = 25,
		    umbrella = 26, handbag = 27, tie = 28, suitcase = 29, frisbee = 30,
		    skis = 31, snowboard = 32, sports ball = 33, kite = 34, baseball bat = 35,
		    baseball glove = 36, skateboard = 37, surfboard = 38, tennis racket = 39, bottle = 40,
		    wine glass = 41, cup = 42, fork = 43, knife = 44, spoon = 45,
		    bowl = 46, banana = 47, apple = 48, sandwich = 49, orange = 50,
		    broccoli = 51, carrot = 52, hot dog = 53, pizza = 54, donut = 55,
		    cake = 56, chair = 57, couch = 58, potted plant = 59, bed = 60,
		    dining table = 61, toilet = 62, tv = 63, laptop = 64, mouse = 65,
		    remote = 66, keyboard = 67, cell phone = 68, microwave = 69, oven = 70,
		    toaster = 71, sink = 72, refrigerator = 73, book = 74, clock = 75,
		    vase = 76, scissors = 77, teddy bear = 78, hair drier = 79, toothbrush = 80
	*/

	private:
	
		PyObject *pInstance, *pModule, *pArgs;

		std::vector<cv::Rect> mvClassBox;
		std::vector<cv::Mat> mvClassMask;
		std::vector<int> mvClassId;
		std::vector<double> mvClassScores;

		std::mutex mMutexNewImages;
		std::mutex mMutexMaskRCNN;
    	std::list<cv::Mat> mlNewImages;
		cv::Mat mCurrentImage;

	public:

		/**	Default constructor
		*/
		MaskRCNN();

		/**	Default destructor
		*/
		~MaskRCNN();

		/** Returns a pointer to a PyObject. A PyObject can represent any Python object.
		*/
		PyObject* getPyObject(const char* name);

		/**	Extracts mask class identification number from maskrcnn-benchmark instance
		@param result A vector of masks label Id
		*/
		void extractClassIDs(std::vector<int>& result);
		void extractClassIDs();

		/**	Extracts mask classification score from maskrcnn-benchmark instance
		@param result A vector of masks class score
		*/
		void extractClassScores(std::vector<double>& result);
		void extractClassScores();

		/**	Extracts object instance bounding box coordinates from maskrcnn-benchmark instance
		@param result A vector of object instance bounding box coordinates
		*/
		void extractBoundingBoxes(std::vector<cv::Rect>& result);
		void extractBoundingBoxes();

		/**	Extracts binary mask images from maskrcnn-benchmark instance
		@param result A vector of binary mask images
		*/
		void extractImage(std::vector<cv::Mat>& result);
		void extractImage();

		/**	Extracts mask class identification number from maskrcnn-benchmark instance
		@param Image input RGB image
		@param Rec vector of bounding box coordinates (one per object instance)
		@param Mask vector of binary mask images (one per object instance)
		@param Id vector of label Ids (one per object instance)
		@param score vector of masks score (one per object instance)
		*/
		void Run(cv::Mat Image, std::vector<cv::Rect>& Rec, std::vector<cv::Mat>& Mask, 
					std::vector<int>& Id, std::vector<double>& score);
		void Run();
		void Run(cv::Mat Image);

		std::vector<int> GetLabels();
		std::vector<double> GetScores();
		std::vector<cv::Mat> GetMasks();
		std::vector<cv::Rect> GetBoxes();

		void InsertImage(cv::Mat Image);
		bool CheckNewImages();
		void ProcessNewImage();
		
		void Reset();

		void show2d(cv::Mat& imRGB, unordered_map<int,string>& categories);

	private:

		/*	
		* Intialise an instance of MaskRCNN
		* This includes setting up the Python environment and loading an instance of its PyObject
		*/
		void initialise();

		/*	
		* Create a python binding instance
		* Requirements: python file and class names and arguments
		*/
		void loadModule();
		
};

#endif // MaskRCNN_H
