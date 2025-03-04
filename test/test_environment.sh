# For MrT gannet
# Check active GPU
gpu=$(nvidia-smi -L)
if [[ $gpu != 0 ]]; then
    gpu=$(echo $gpu | cut -b 1-19)
	echo "Active" $gpu
else
	echo "No Active GPU"
fi

# Check internet connection
wget --spider --quiet http://google.com
if [ "$?" != O ]; then
	echo "With Internet Connection"
else
	echo "No Internet Connection"
fi

echo "Environmental variables included: "
# add paths to ~/.bashrc

# Cuda-9-0
if [ -d "/usr/local/cuda-9.0/bin" ] ; then
	PATH="$PATH:/usr/local/cuda-9.0/bin"
	echo " + Found CUDA 9.0/bin"
else
	echo " + Not Found CUDA 9.0/bin at /usr/local/cuda-9.0/bin"
fi
if [ -d "/usr/local/cuda-9.0/lib64" ] ; then
	LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64"
	echo " + Found CUDA 9.0/lib64"
else
	echo " + Not Found CUDA 9.0/lib64 at /usr/local/cuda-9.0/lib64"
fi

# cuDNN 7.0.5
if [ -d "/usr/local/cuda/extras/CUPTI/lib64" ] ; then
	LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"
	echo " + Found cuDNN 7.0.5/lib64"
else
	echo " + Not Found cuDNN 7.0.5/lib64 at /usr/local/cuda/extras/CUPTI/lib64"
fi

cppcompiler=$(gcc --version)
cppcompiler=$(echo $cppcompiler | cut -b 13-17)
echo " + Found gcc" $cppcompiler
cppcompiler=$(g++ --version)
cppcompiler=$(echo $cppcompiler | cut -b 13-17)
echo " + Found g++" $cppcompiler
echo " + Found opencv" $(pkg-config --modversion opencv)
#echo " - "$(python --version)
echo " - Found "$(python3 --version)
echo "          matplotlib" $(python3 -c "import matplotlib; print(matplotlib.__version__)")
echo "          numpy" $(python3 -c "import numpy; print(numpy.__version__)")
echo "          pillow" $(python3 -c "import PIL; print(PIL.__version__)")
echo "          torch" $(python3 -c "import torch; print(torch.__version__)")
echo "          torchvision" $(python3 -c "import torchvision; print(torchvision.__version__)")
echo "          cv2" $(python3 -c "import cv2; print(cv2.__version__)")