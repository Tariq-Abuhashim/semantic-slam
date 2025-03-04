
/*
 * @file MaskRCNN.cpp
 * This is part of Semantic SLAM.
 * An interface to run Mask-RCNN in Python.
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-07-00
 */

#include "MaskRCNN.hpp"
#include <iostream>

typedef std::vector<std::vector<cv::Point> > Contour;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define DEBUG 0

using namespace std;

/*	
*	Construct a python instance
*/
MaskRCNN::MaskRCNN( ): pInstance(NULL) 
{
	initialise();
}

/*	
*	Destroy the python instance
*/
MaskRCNN::~MaskRCNN() 
{
	Py_XDECREF(pModule);
    Py_XDECREF(pInstance);
	Py_Finalize();
	cout << "MaskRCNN has been deleted" << endl;
}

/*	
* 
*/
void MaskRCNN::initialise() 
{

	cout << " * Initialising MaskRCNN ... ";
	Py_SetProgramName((wchar_t*)L"MaskRCNN");
    Py_Initialize();
	wchar_t const * argv2[] = { L"MaskRCNN.py" };
    PySys_SetArgv(1, const_cast<wchar_t**>(argv2));

	// Load module
	loadModule();

/*
	// Get function in one shot, this assumes no class in MaskRCNN.py
	// If MaskRCNN.py contains classes, then check ~/Dev/cpp_to_py for appropriate wrapper
    pInstance = PyObject_GetAttrString(pModule, "predict");
    if(pInstance == NULL || !PyCallable_Check(pInstance)) {
        if(PyErr_Occurred()) {
            cerr << endl << "Python error indicator is set:" << endl;
            PyErr_Print();
        }
        throw runtime_error("Could not load function 'predict' from MaskRCNN module.");
    }
*/

	cout << "Done !" << endl;

}

/*	
* 
*/
void MaskRCNN::loadModule() 
{
	//cout << " * Loading module..." << endl;
	pModule = PyImport_ImportModule("MaskRCNN");
	if(pModule == NULL) {
        if(PyErr_Occurred()) {
            cerr << endl << "Python error indicator is set:" << endl;
            PyErr_Print();
        }
        throw runtime_error("Could not open MaskRCNN module.");
    }
	//import_array(); // Mask-Fusion way
	//return 0; // Mask-Fusion way
}

/*	
*	Generate a python object for the maskrcnn function to be called
*/
PyObject* MaskRCNN::getPyObject(const char* name)
{
    PyObject* obj = PyObject_GetAttrString(pModule, name);
    if(!obj || obj == Py_None) throw runtime_error(string("Failed to get python object: ") + name);
    return obj;
}

/*	
* 
*/
void MaskRCNN::extractClassIDs(vector<int>& result)
{
    assert(result.size() == 0);
    PyObject* pClassList = getPyObject("labels");
    if(!PySequence_Check(pClassList)) throw runtime_error("pClassList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pClassList);
    result.reserve(n);
    //result->push_back(0); // Background
    for (int i = 0; i < n; ++i) {
        PyObject* o = PySequence_GetItem(pClassList, i);
        assert(PyLong_Check(o));
        result.push_back(PyLong_AsLong(o));
        Py_DECREF(o);
    }
    Py_DECREF(pClassList);
}

/*	
* 
*/
void MaskRCNN::extractClassIDs()
{

	unique_lock<mutex> lock(mMutexMaskRCNN);

    assert(result.size() == 0);
    PyObject* pClassList = getPyObject("labels");
    if(!PySequence_Check(pClassList)) throw runtime_error("pClassList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pClassList);

    mvClassId.reserve(n); // FIXME: add to .hpp

    //result->push_back(0); // Background
    for (int i = 0; i < n; ++i) {
        PyObject* o = PySequence_GetItem(pClassList, i);
        assert(PyLong_Check(o));

        mvClassId.push_back(PyLong_AsLong(o));

        Py_DECREF(o);
    }
    Py_DECREF(pClassList);
}

/*	
* 
*/
void MaskRCNN::extractClassScores(vector<double>& result) 
{
	assert(result.size() == 0);
    PyObject* pClassList = getPyObject("scores");
    if(!PySequence_Check(pClassList)) throw runtime_error("pClassList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pClassList);
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
        PyObject* o = PySequence_GetItem(pClassList, i);
        assert(PyFloat_Check(o));
        result.push_back(PyFloat_AsDouble(o));
        Py_DECREF(o);
    }
    Py_DECREF(pClassList);
}

void MaskRCNN::extractClassScores() 
{
	unique_lock<mutex> lock(mMutexMaskRCNN);

	assert(result.size() == 0);
    PyObject* pClassList = getPyObject("scores");
    if(!PySequence_Check(pClassList)) throw runtime_error("pClassList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pClassList);

    mvClassScores.reserve(n); // FIXME: add to .hpp

    for (int i = 0; i < n; ++i) {
        PyObject* o = PySequence_GetItem(pClassList, i);
        assert(PyFloat_Check(o));

        mvClassScores.push_back(PyFloat_AsDouble(o));

        Py_DECREF(o);
    }
    Py_DECREF(pClassList);
}

/*	
* 
*/
void MaskRCNN::extractBoundingBoxes(vector<cv::Rect>& result){
    assert(result.size() == 0);
    PyObject* pRoiList = getPyObject("boxes");
    if(!PySequence_Check(pRoiList)) throw runtime_error("pRoiList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pRoiList);
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
        PyObject* pRoi = PySequence_GetItem(pRoiList, i);
        assert(PySequence_Check(pRoi));
        assert(PySequence_Length(pRoi)==4);

        PyObject* c0 = PySequence_GetItem(pRoi, 0);
        PyObject* c1 = PySequence_GetItem(pRoi, 1);
        PyObject* c2 = PySequence_GetItem(pRoi, 2);
        PyObject* c3 = PySequence_GetItem(pRoi, 3);
        assert(PyLong_Check(c0) && PyLong_Check(c1) && PyLong_Check(c2) && PyLong_Check(c3));

        int a = PyLong_AsLong(c0);
        int b = PyLong_AsLong(c1);
        int c = PyLong_AsLong(c2);
        int d = PyLong_AsLong(c3);

        Py_DECREF(c0);
        Py_DECREF(c1);
        Py_DECREF(c2);
        Py_DECREF(c3);

        result.push_back(cv::Rect(a,b,c-a,d-b));
        Py_DECREF(pRoi);
    }
    Py_DECREF(pRoiList);
}

/*	
* 
*/
void MaskRCNN::extractBoundingBoxes()
{

	unique_lock<mutex> lock(mMutexMaskRCNN);

    assert(result.size() == 0);
    PyObject* pRoiList = getPyObject("boxes");
    if(!PySequence_Check(pRoiList)) throw runtime_error("pRoiList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pRoiList);

    mvClassBox.reserve(n); // FIXME: add to .hpp

    for (int i = 0; i < n; ++i) {
        PyObject* pRoi = PySequence_GetItem(pRoiList, i);
        assert(PySequence_Check(pRoi));
        assert(PySequence_Length(pRoi)==4);

        PyObject* c0 = PySequence_GetItem(pRoi, 0);
        PyObject* c1 = PySequence_GetItem(pRoi, 1);
        PyObject* c2 = PySequence_GetItem(pRoi, 2);
        PyObject* c3 = PySequence_GetItem(pRoi, 3);
        assert(PyLong_Check(c0) && PyLong_Check(c1) && PyLong_Check(c2) && PyLong_Check(c3));

        int a = PyLong_AsLong(c0);
        int b = PyLong_AsLong(c1);
        int c = PyLong_AsLong(c2);
        int d = PyLong_AsLong(c3);

        Py_DECREF(c0);
        Py_DECREF(c1);
        Py_DECREF(c2);
        Py_DECREF(c3);

        mvClassBox.push_back(cv::Rect(a,b,c-a,d-b));

        Py_DECREF(pRoi);
    }
    Py_DECREF(pRoiList);
}

/*	
* 
*/
void MaskRCNN::extractImage(vector<cv::Mat>& result){

	PyObject* pMaskList = getPyObject("masks");
    if(!PySequence_Check(pMaskList)) // Returns 1 if the object provides sequence protocol, and 0 otherwise.
		throw runtime_error("pMaskList is not a sequence.");
	Py_ssize_t n = PySequence_Length(pMaskList);
	result.reserve(n);

/*
	PyObject* pImage = getPyObject("result");
    PyArrayObject *pImageArray = (PyArrayObject*)(pImage);
    unsigned char* pData = (unsigned char*)PyArray_DATA(pImageArray);
    npy_intp h = PyArray_DIM(pImageArray,0);
    npy_intp w = PyArray_DIM(pImageArray,1);
	cv::Mat image;
	cv::Mat(h, w, CV_16UC3, pData).copyTo(image);
	result.push_back(255*image);
	Py_DECREF(pImage);
*/

    for (int i = 0; i < n; ++i) {
		PyObject* pMask = PySequence_GetItem(pMaskList, i);
		assert(PySequence_Check(pMask));

		PyArrayObject *pMaskArray = (PyArrayObject*)(pMask);
		assert(pMaskArray->flags & NPY_ARRAY_C_CONTIGUOUS);

		unsigned char* pData = (unsigned char*)PyArray_DATA(pMaskArray);
		npy_intp h = PyArray_DIM(pMaskArray,1);
		npy_intp w = PyArray_DIM(pMaskArray,2);

		cv::Mat mask;
		cv::Mat(h,w, CV_8UC1, pData).copyTo(mask);

		result.push_back(255*mask);

		Py_DECREF(pMask);
	}

}

/*	
* 
*/
void MaskRCNN::extractImage()
{

	unique_lock<mutex> lock(mMutexMaskRCNN);

	PyObject* pMaskList = getPyObject("masks");
    if(!PySequence_Check(pMaskList)) // Returns 1 if the object provides sequence protocol, and 0 otherwise.
		throw runtime_error("pMaskList is not a sequence.");
	Py_ssize_t n = PySequence_Length(pMaskList);

	mvClassMask.reserve(n); // FIXME: add to .hpp

/*
	PyObject* pImage = getPyObject("result");
    PyArrayObject *pImageArray = (PyArrayObject*)(pImage);
    unsigned char* pData = (unsigned char*)PyArray_DATA(pImageArray);
    npy_intp h = PyArray_DIM(pImageArray,0);
    npy_intp w = PyArray_DIM(pImageArray,1);
	cv::Mat image;
	cv::Mat(h, w, CV_16UC3, pData).copyTo(image);
	cv::imshow("maskrcnn", image);
    cv::waitKey(0);
	result.push_back(255*image);
	Py_DECREF(pImage);
*/

    for (int i = 0; i < n; ++i) 
	{
		PyObject* pMask = PySequence_GetItem(pMaskList, i);
		assert(PySequence_Check(pMask));

		PyArrayObject *pMaskArray = (PyArrayObject*)(pMask);
		assert(pMaskArray->flags & NPY_ARRAY_C_CONTIGUOUS);

		unsigned char* pData = (unsigned char*)PyArray_DATA(pMaskArray);
		npy_intp h = PyArray_DIM(pMaskArray,1);
		npy_intp w = PyArray_DIM(pMaskArray,2);

		cv::Mat mask;
		cv::Mat(h,w, CV_8UC1, pData).copyTo(mask);

		mvClassMask.push_back(255*mask);

		Py_DECREF(pMask);
	}

}

/*	
* 
*/
void MaskRCNN::Run()
{
	while(1)
    {
        // Tracking will see that Local Mapping is busy
        //SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewImages())
        {
            ProcessNewImage();
		}
	}
}

/*	
* 
*/
void MaskRCNN::InsertImage(cv::Mat Image)
{
    unique_lock<mutex> lock(mMutexNewImages);
    mlNewImages.push_back(Image);
}

/*	
* 
*/
bool MaskRCNN::CheckNewImages()
{
    unique_lock<mutex> lock(mMutexNewImages);
    return(!mlNewImages.empty());
}

/*	
* 
*/
void MaskRCNN::ProcessNewImage()
{
	Reset();

    {
        unique_lock<mutex> lock(mMutexNewImages);
        mCurrentImage = mlNewImages.front();
        mlNewImages.pop_front();
    }

	// Get function in one shot, this assumes no class in MaskRCNN.py
	// If MaskRCNN.py contains classes, then check ~/Dev/cpp_to_py for appropriate wrapper
    pInstance = PyObject_GetAttrString(pModule, "predict");
    if(pInstance == NULL || !PyCallable_Check(pInstance)) {
        if(PyErr_Occurred()) {
            cerr << endl << "Python error indicator is set:" << endl;
            PyErr_Print();
        }
        throw runtime_error("Could not load function 'predict' from MaskRCNN module.");
    }

	_import_array();

	int depth = CV_MAT_DEPTH(mCurrentImage.type());
	//int cn = CV_MAT_CN(Image.type());
	const int f = (int)(sizeof(size_t)/8);
 	int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                  depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                  depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                  depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;

	npy_intp dims[3] = {mCurrentImage.rows, mCurrentImage.cols, mCurrentImage.channels()};
	pArgs = PyArray_SimpleNewFromData(mCurrentImage.dims+1, dims, typenum, mCurrentImage.data);

	// Run MaskRCNN
	Py_XDECREF(PyObject_CallFunctionObjArgs(pInstance, pArgs, NULL));

	// Extract information from MaskRCNN object
	extractClassIDs();
	extractClassScores();
	extractBoundingBoxes();
	extractImage();

}

/*	
*	Run instance from input image cv::Mat
*	Requirements: python function name and arguments
*/
void MaskRCNN::Run(cv::Mat Image, vector<cv::Rect>& vClassBox, vector<cv::Mat>& vClassMask, vector<int>& vClassId, vector<double>& vClassScores) {
	//cout << " -- C++: using Mat" << endl;
	Reset();

	// Get function in one shot, this assumes no class in MaskRCNN.py
	// If MaskRCNN.py contains classes, then check ~/Dev/cpp_to_py for appropriate wrapper
    pInstance = PyObject_GetAttrString(pModule, "predict");
    if(pInstance == NULL || !PyCallable_Check(pInstance)) {
        if(PyErr_Occurred()) {
            cerr << endl << "Python error indicator is set:" << endl;
            PyErr_Print();
        }
        throw runtime_error("Could not load function 'predict' from MaskRCNN module.");
    }

	//assert(rgbImage.channels() == 3);
	_import_array();

	int depth = CV_MAT_DEPTH(Image.type());
	//int cn = CV_MAT_CN(Image.type());
	const int f = (int)(sizeof(size_t)/8);
 	int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                  depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                  depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                  depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
	
	//cout << "h: "<<Image.rows << ", w: "<< Image.cols << ", c: "<< Image.channels() << ", d: "<< typenum << endl;

	npy_intp dims[3] = {Image.rows, Image.cols, Image.channels()};
	pArgs = PyArray_SimpleNewFromData(Image.dims+1, dims, typenum, Image.data);

	// Run MaskRCNN
	Py_XDECREF(PyObject_CallFunctionObjArgs(pInstance, pArgs, NULL));

	// Extract information from MaskRCNN object

	//vector<int>* vClassId = new vector<int>;
	extractClassIDs(vClassId);
#if DEBUG
	cout << endl << "- Class Id:" << endl;
	for(vector<int>::iterator itr=vClassId.begin(); itr<vClassId.end(); itr++) 
		cout << "     " << *itr << endl;
#endif

	extractClassScores(vClassScores);

	//vector<cv::Rect> vClassBox;
	extractBoundingBoxes(vClassBox);
#if DEBUG
	cout << endl << "- Class Box <y1, x1, y2, x2>:" << endl;
	for(vector<cv::Rect>::iterator itr=vClassBox.begin(); itr<vClassBox.end(); itr++) {
		cv::Rect bbox = *itr;
		cout << "     <" << bbox.y << ", " << bbox.x << ", " << bbox.height+bbox.y << ", " << bbox.width+bbox.x << ">" << endl;
	}
#endif

	//vector<cv::Mat>* vClassMask = new vector<cv::Mat>;
	extractImage(vClassMask);
#if DEBUG
	cout << endl << "- vClassMask.size() " << vClassMask.size() << endl;
	int i = 0;
	for(vector<cv::Mat>::iterator itr=vClassMask.begin(); itr<vClassMask.end(); itr++) {
		cv::Mat mask = *itr;
		//cout << "     <" << mask.rows << ", " << mask.cols << ">" << endl;
		string file_name = "mask" + to_string(i++) + ".png";
		cv::imwrite(file_name , mask );
	}
#endif
}

/*	
*	Run instance from input image cv::Mat
*	Requirements: python function name and arguments
*/
void MaskRCNN::Run(cv::Mat Image) {
	//cout << " -- C++: using Mat" << endl;
	Reset();

	// Get function in one shot, this assumes no class in MaskRCNN.py
	// If MaskRCNN.py contains classes, then check ~/Dev/cpp_to_py for appropriate wrapper
    pInstance = PyObject_GetAttrString(pModule, "predict");
    if(pInstance == NULL || !PyCallable_Check(pInstance)) {
        if(PyErr_Occurred()) {
            cerr << endl << "Python error indicator is set:" << endl;
            PyErr_Print();
        }
        throw runtime_error("Could not load function 'predict' from MaskRCNN module.");
    }

	//assert(rgbImage.channels() == 3);
	_import_array();

	int depth = CV_MAT_DEPTH(Image.type());
	//int cn = CV_MAT_CN(Image.type());
	const int f = (int)(sizeof(size_t)/8);
 	int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                  depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                  depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                  depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
	
	//cout << "h: "<<Image.rows << ", w: "<< Image.cols << ", c: "<< Image.channels() << ", d: "<< typenum << endl;

	npy_intp dims[3] = {Image.rows, Image.cols, Image.channels()};
	pArgs = PyArray_SimpleNewFromData(Image.dims+1, dims, typenum, Image.data);

	// Run MaskRCNN
	Py_XDECREF(PyObject_CallFunctionObjArgs(pInstance, pArgs, NULL));

	// Extract information from MaskRCNN object

	extractClassIDs();
#if DEBUG
	cout << endl << "- Class Id:" << endl;
	for(vector<int>::iterator itr=vClassId.begin(); itr<vClassId.end(); itr++) 
		cout << "     " << *itr << endl;
#endif

	extractClassScores();

	extractBoundingBoxes();
#if DEBUG
	cout << endl << "- Class Box <y1, x1, y2, x2>:" << endl;
	for(vector<cv::Rect>::iterator itr=mvClassBox.begin(); itr<mvClassBox.end(); itr++) {
		cv::Rect bbox = *itr;
		cout << "     <" << bbox.y << ", " << bbox.x << ", " << bbox.height+bbox.y << ", " << bbox.width+bbox.x << ">" << endl;
	}
#endif

	extractImage();
#if DEBUG
	cout << endl << "- vClassMask.size() " << mvClassMask.size() << endl;
	int i = 0;
	for(vector<cv::Mat>::iterator itr=mvClassMask.begin(); itr<mvClassMask.end(); itr++) {
		cv::Mat mask = *itr;
		//cout << "     <" << mask.rows << ", " << mask.cols << ">" << endl;
		string file_name = "mask" + to_string(i++) + ".png";
		cv::imwrite(file_name , mask );
	}
#endif
}

std::vector<int> MaskRCNN::GetLabels()
{
	return mvClassId;
}
std::vector<double> MaskRCNN::GetScores()
{
	return mvClassScores;
}
std::vector<cv::Mat> MaskRCNN::GetMasks()
{
	return mvClassMask;
}
std::vector<cv::Rect> MaskRCNN::GetBoxes()
{
	return mvClassBox;
}

void MaskRCNN::Reset()
{
	mvClassId.clear(); mvClassId.resize(0);
	mvClassScores.clear(); mvClassScores.resize(0);
	mvClassMask.clear(); mvClassMask.resize(0);
	mvClassBox.clear(); mvClassBox.resize(0);
}


void MaskRCNN::show2d(cv::Mat& imRGB, unordered_map<int,string>& categories)
{

	cv::Mat image = imRGB.clone();

	for(size_t i=0; i<mvClassMask.size(); i++)
	{
		cv::Mat mask = mvClassMask[i];
		Contour maskContour;
		std::vector<cv::Vec4i> hierarchy;
		findContours( mask, maskContour, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

		cv::Rect bbox = mvClassBox[i];
		std::string score = std::to_string(mvClassScores[i]);
		std::string label = categories[mvClassId[i]];
		cv::Scalar color = cv::Scalar(std::rand()%255, std::rand()%255, std::rand()%255);
		cv::drawContours( image, maskContour, -1, color, 1, 4, hierarchy, 0, cv::Point() );
		cv::putText(image, score, cv::Point(bbox.x+2,bbox.y-2), cv::FONT_HERSHEY_DUPLEX, 0.5, color, 1);
		cv::putText(image, label, cv::Point(bbox.x+2,bbox.y-15), cv::FONT_HERSHEY_DUPLEX, 0.5, color, 1);
	}

	cv::namedWindow( "Texture Segmentation", CV_WINDOW_AUTOSIZE );
	cv::imshow( "Texture Segmentation", image );

	std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(1);
	cv::imwrite("maskrcnn.png", image, compression_params);

	cv::waitKey(100);

}


//void MaskRCNN::loop(){
//}
//void MaskRCNN::startThreadLoop(){
//}


