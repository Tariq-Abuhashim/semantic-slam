
#ifndef INVENTORY_H
#define INVENTORY_H

#include "Object.hpp"
#include "Thirdparty/ORB_SLAM2/include/KeyFrame.h"

#include <unordered_set>
#include <mutex>

/** @brief An inventory of all detected and classified map objects
	@author Tariq Abuhashim
	@date August 2019
*/
class Inventory
{

	public:

	/** Default constructor
	*/
	Inventory();

	~Inventory();

	/**	Add a new detected and classified object into the inventory
	@param MO the current MapObject to be added to inventory
	*/
	void AddObject(Object* O);

	/**	Return a pointer to the object with index (idx)
	@param idx index to object.
	*/
	Object* GetObject(int idx);

	/**	Return a vector of all objects in the inventory
	*/
	vector<Object*> GetAllObjects();

	void AddKeyFrame(ORB_SLAM2::KeyFrame *KF);
	vector<ORB_SLAM2::KeyFrame*> GetAllKeyFrames();

	private:

	std::mutex mMutexInventory;
	unordered_set<Object*> mspObjects;
	unordered_set<ORB_SLAM2::KeyFrame*> mspKeyFrames;

};

#endif
