
CL='\033[1;91m'
NC='\033[0m' # No Color

cd Thirdparty/

# install pycocotools
echo -e "${CL}Configuring and building Thirdparty/cocoapi ...${NC}"
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
sudo python3 setup.py build_ext install
cd ../../

# install apex
echo -e "${CL}Configuring and building Thirdparty/apex ...${NC}"
git clone https://github.com/NVIDIA/apex.git
cd apex
sudo python3 setup.py install --cuda_ext --cpp_ext
cd ../

# install PyTorch Detection (maskrcnn-benchmark)
echo -e "${CL}Configuring and building Thirdparty/maskrcnn-benchmark ...${NC}"
https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
sudo python3 setup.py build develop
cd ../

# install PyTorch Detection (Detectron2)
#sudo pip3 install 'git+https://github.com/facebookresearch/fvcore'
##sudo pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
#git clone https://github.com/facebookresearch/detectron2.git
#cd detectron2
#python3 setup.py build develop

# install ORB_SLAM2
echo -e "${CL}Configuring and building Thirdparty/ORB_SLAM2 ...${NC}"
git clone https://github.com/raulmur/ORB_SLAM2.git
scp ../include/ORB_SLAM2/System.h ./ORB_SLAM2/include # change some files
scp ../src/ORB_SLAM2/System.cc ./ORB_SLAM2/src # change some files
cd ORB_SLAM2
./build.sh
cd ../../

# install semantic SLAM example
echo -e "${CL}Configuring and building Semantic SLAM Example ...${NC}"
mkdir build
cd build
cmake .. #-DCMAKE_CUDA_FLAGS=”-arch=sm_50” -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..  # cuda flag to compile against cv 2.4.13
make
cd ../

# build the documentation
echo -e "${CL}Building documentation ...${NC}"
pwd
cd doxygen
doxygen doc.config
cd ../
ln -s ./doxygen/html/index.html README.html
