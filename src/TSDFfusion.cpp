
/*
 * @file Inventory.cpp
 * This is part of Semantic SLAM.
 * Functions to handle a 3D object TSDF.
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-08-00
 */


#include "TSDFfusion.hpp"
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define DEBUG 0

using namespace std;

/*	
*	Construct a python instance
*/
TSDFfusion::TSDFfusion( ): pInstance(NULL) {
	initialise();
}

/*	
*	Destroy the python instance
*/
TSDFfusion::~TSDFfusion() {
	Py_XDECREF(pModule);
    Py_XDECREF(pInstance);
	Py_Finalize();
	cout << "TSDFfusion has been deleted." << endl;
}

/*	
* 
*/
void TSDFfusion::initialise() {

	cout << " * Initialising TSDFfusion ... ";
	Py_SetProgramName((wchar_t*)L"TSDFfusion");
    Py_Initialize();
	wchar_t const * argv2[] = { L"TSDFfusion.py" };
    PySys_SetArgv(1, const_cast<wchar_t**>(argv2));

	// Load module
	loadModule();

	cout << "Done !" << endl;
}

/*	
* 
*/
void TSDFfusion::loadModule() {
	//cout << " * Loading module..." << endl;
	pModule = PyImport_ImportModule("TSDFfusion");
	if(pModule == NULL) {
        if(PyErr_Occurred()) {
            cerr << endl << "Python error indicator is set:" << endl;
            PyErr_Print();
        }
        throw runtime_error("Could not open MaskRCNN module.");
    }
}

/*	
*	Requirements: python function name and arguments
*/
void TSDFfusion::Integrate(cv::Mat imRGB, cv::Mat imD) {
	
	// Get function in one shot, this assumes no class in TSDFfusion.py
	// If TSDFfusion.py contains classes, then check ~/Dev/cpp_to_py for appropriate wrapper
    pInstance = PyObject_GetAttrString(pModule, "Integrate");
    if(pInstance == NULL || !PyCallable_Check(pInstance)) {
        if(PyErr_Occurred()) {
            cerr << endl << "Python error indicator is set:" << endl;
            PyErr_Print();
        }
        throw runtime_error("Could not load function 'Integrate' from TSDFfusion module.");
    }

	_import_array();

	int depth = CV_MAT_DEPTH(imRGB.type());
	const int f = (int)(sizeof(size_t)/8);
 	int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                  depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                  depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                  depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;

	npy_intp dims1[3] = {imRGB.rows, imRGB.cols, imRGB.channels()};
	pArgs1 = PyArray_SimpleNewFromData(imRGB.dims+1, dims1, typenum, imRGB.data);

	npy_intp dims2[3] = {imD.rows, imD.cols, imD.channels()};
	pArgs2 = PyArray_SimpleNewFromData(imD.dims+1, dims2, typenum, imD.data);

	// Run Integrate in Python
	Py_XDECREF(PyObject_CallFunctionObjArgs(pInstance, pArgs1, pArgs2, NULL));

}


