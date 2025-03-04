
/*
*
* Tariq Abuhashim
* t.abuhashim@gmail.com
* August, 2019
*
*/

#ifndef TSDFfusion_H
#define TSDFfusion_H

#include <stdio.h>
#include <Python.h>
#include <unicodeobject.h>
#include <opencv2/opencv.hpp>

#include <unordered_map>
#include <string>

/** @brief An interface between TSDF-fusion-python in python and Engine in C++
	@author Tariq Abuhashim
	@date August 2019
*/
class TSDFfusion 
{

	private:
	
		PyObject *pInstance, *pModule, *pArgs1, *pArgs2;

	public:

		/**	Default constructor
		*/
		TSDFfusion();

		/**	Default destructor
		*/
		~TSDFfusion();

		/**	Extracts mask class identification number from maskrcnn-benchmark instance
		@param Image input RGB image
		@param Rec vector of bounding box coordinates (one per object instance)
		@param Mask vector of binary mask images (one per object instance)
		@param Id vector of label Ids (one per object instance)
		@param score vector of masks score (one per object instance)
		*/
		void Integrate(cv::Mat imRGB, cv::Mat imD);
		
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

#endif // TSDFfusion_H
