
/*
*
* Tariq Abuhashim
* t.abuhashim@gmail.com
* July, 2019
*
*/

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


int main () 

{
	ifstream fCoords;
	fCoords.open ("/home/mrt/Dev/semantic_slam_b/result/coords.txt");
	if (!fCoords.is_open()) return 1;

	string s;

	while (getline(fCoords, s)) 
	{
		
		size_t index = s.find("#index = "); // first occurance
		if (index != string::npos) 
		{
			size_t comma = s.find(",", index+1);
			if (comma != string::npos) 
			{
				string ncam(s,index+9,comma-(index+9));
				cout << ncam << " ";
			}
		}

		size_t npts;

		size_t keys = s.find("keys = "); // first occurance
		if (keys != string::npos) 
		{
			size_t comma = s.find(",", keys+1);
			if (comma != string::npos) 
			{
				string npts_str(s,keys+7,comma-(keys+7));
				stringstream ss(npts_str);
				ss >> npts;
				cout << npts << endl;
			}
		}

		for(int i=0; i<npts; i++)
		{
			getline(fCoords, s);
			stringstream ss;
			ss << s;
			size_t KPId;
			ss >> KPId;
			float x, y;
			ss >> x;
			ss >> y;
			cout << KPId << " " << x << " " << y << endl;
		}
	}

}
