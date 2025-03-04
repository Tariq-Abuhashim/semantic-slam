#ifndef CLOUDVIEWER_H
#define CLOUDVIEWER_H

#include <pcl/PointIndices.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "CloudViewer.hpp"
#include <pangolin/pangolin.h>

template<class T>
class CloudViewer 
{

	private:
		
		/// DoN map computed using DoN operator (equation 2).
		typename pcl::PointCloud<T>::Ptr mpCloud;

		/// Segmentation of the DoN map based on threshold over DoN magnitude.
		std::vector<pcl::PointIndices> mvClusters;

		float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
		bool bHasIntensity;

	private:
	
		float RandomFloat()
		{
			return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}

		void DrawClusters()
		{

		}

		void DrawCloud()
		{
			size_t cloudSize = mpCloud->points.size();
			glPointSize(2);
			glBegin(GL_POINTS);
			glColor3f(0,0,0);

			

			for(size_t i=0; i<cloudSize; i++)
			{
				//if(bHasIntensity)
				//glColor3f(mpCloud->points[i].intensity, mpCloud->points[i].intensity, mpCloud->points[i].intensity);
				std::cout <<mpCloud->points[i].x<<" "<<mpCloud->points[i].y<<" "<<mpCloud->points[i].z<< std::endl;
				glVertex3f(mpCloud->points[i].x,mpCloud->points[i].y,mpCloud->points[i].z);
			}
			glEnd();
		}

	public:

		CloudViewer( typename pcl::PointCloud<T>::Ptr Cloud, std::vector<pcl::PointIndices> Clusters) : 
			mpCloud(Cloud), mvClusters(Clusters), mViewpointX(0), mViewpointY(-0.1), mViewpointZ(-100), 
			mViewpointF(1000)
		{
			std::cout << "Cloud viewer ready." << std::endl;
		}

		CloudViewer( typename pcl::PointCloud<T>::Ptr Cloud): 
		mpCloud(Cloud), mViewpointX(0), mViewpointY(-0.1), mViewpointZ(-100), mViewpointF(1000)
		{
			std::cout << "Cloud viewer ready." << std::endl;
		}


		void Run()
		{

			pangolin::CreateWindowAndBind("Cloud viewer",1024,768);

			// 3D Mouse handler requires depth testing to be enabled
			glEnable(GL_DEPTH_TEST);

			// Issue specific OpenGl we might need
			glEnable (GL_BLEND);
			glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			// Add save button
			pangolin::CreatePanel("gui").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
			pangolin::Var<bool> menuSave("gui.Save",false,false);

			// Define Camera Render Object (for view / scene browsing)
			pangolin::OpenGlRenderState s_cam(
				        pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
				        pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0, 0,0,-1.0)
				        );

			// Add named OpenGL viewport to window and provide 3D Handler
			pangolin::View& d_cam = pangolin::CreateDisplay()
				    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
				    .SetHandler(new pangolin::Handler3D(s_cam));

			pangolin::OpenGlMatrix Twc;
			Twc.SetIdentity();

			//while(pangolin::ShouldQuit()==false)
			std::vector<std::vector<float>> colors;
			for(std::vector<pcl::PointIndices>::const_iterator it=mvClusters.begin();
					it!=mvClusters.end();++it) {
					std::vector<float> color;
					color.push_back(RandomFloat());
					color.push_back(RandomFloat());
					color.push_back(RandomFloat());
					colors.push_back(color);
			}

			while(1)
			{

				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				s_cam.Follow(Twc);

				if(menuSave)
				{
					d_cam.SaveOnRender("pangolin");
					menuSave = false;
				}

				d_cam.Activate(s_cam);
				glClearColor(1.0f,1.0f,1.0f,1.0f);

				//if(mvClusters.size()>0 & mpCloud->points.size()>0)
				//	DrawClusters();
				//else if(mpCloud->points.size()>0)
				//	DrawCloud();

				// Rendering function
				glPointSize(2);
				glBegin(GL_POINTS);
				glColor3f(0,0,0);

				int count = 0;
				if (mvClusters.size()<0) {
					for(size_t j=0; j<mpCloud->points.size(); j++) {
						glVertex3f(	mpCloud->points[j].x,
									mpCloud->points[j].y,
									mpCloud->points[j].z );
					}
				} else {
					for(std::vector<pcl::PointIndices>::const_iterator it=mvClusters.begin();
					it!=mvClusters.end();++it) {
						std::vector<float> color = colors[count];
						glColor3f(color[0],color[1],color[2]);
						count++;
						for (std::vector<int>::const_iterator pit = it->indices.begin(); 
						pit!=it->indices.end();++pit) {
							glVertex3f(	mpCloud->points[*pit].x,
										mpCloud->points[*pit].y,
										mpCloud->points[*pit].z );
						}
					}
				}
				
				glEnd();
				pangolin::FinishFrame();

				usleep(5000);   // sleep 5 ms
			}

		}


};

#endif
