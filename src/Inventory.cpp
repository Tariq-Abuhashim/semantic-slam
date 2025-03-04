
/*
 * @file Inventory.cpp
 * This is part of Semantic SLAM.
 * Functions to handle objects inventory
 *
 * @author Tariq Abuhashim (t.abuhashim@gmail.com)
 * @date 2019-07-00
 */

#include "Inventory.hpp"

Inventory::Inventory () 
{
	cout << "Starting a new inventory ..." << endl;
}

Inventory::~Inventory()
{
	unique_lock<mutex> lock(mMutexInventory);
	int count = 0;
	for(auto sit=mspObjects.begin(); sit!=mspObjects.end(); sit++)
	{
		// object
		Object* object = *sit;
		
		// save object to text file
		object->SaveToFile(std::to_string(count++));

		// delete object pointer
		delete(*sit);
	}
	cout << "Inventory is empty ..." << endl;
}

void Inventory::AddObject(Object* O) 
{
	unique_lock<mutex> lock(mMutexInventory);
	mspObjects.insert(O);
}

vector<Object*> Inventory::GetAllObjects() 
{
	unique_lock<mutex> lock(mMutexInventory);
	return vector<Object*>(mspObjects.begin(),mspObjects.end());
}

Object* Inventory::GetObject(int idx) 
{
	unique_lock<mutex> lock(mMutexInventory);
	vector<Object*> AllObjects = GetAllObjects();
	sort(AllObjects.begin(),AllObjects.end(),Object::lId); // sort Objects according to mnId
	return AllObjects[idx];
}

void Inventory::AddKeyFrame(ORB_SLAM2::KeyFrame *KF)
{
	unique_lock<mutex> lock(mMutexInventory);
	mspKeyFrames.insert(KF);
}

vector<ORB_SLAM2::KeyFrame*> Inventory::GetAllKeyFrames()
{
	unique_lock<mutex> lock(mMutexInventory);
    return vector<ORB_SLAM2::KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

//void CreateNewObject(rows, cols, label_string, score)
//{
//	MapObject*MO = new MapObject(rows, cols);
//	MO->SetLabel(label_string);
//	MO->UpdateScore(score);
//}
