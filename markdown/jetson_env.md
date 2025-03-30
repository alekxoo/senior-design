Current system details: 

machvision@machvision-desktop:~/Documents/sys_diag/jetsonUtilities$ python jetsonInfo.py
NVIDIA NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
 L4T 36.4.3 [ JetPack UNKNOWN ]
   Ubuntu 22.04.5 LTS
   Kernel Version: 5.15.148-tegra
 CUDA 12.6.68
   CUDA Architecture: 8.7
 OpenCV version: 4.8.0
   OpenCV Cuda: NO
 CUDNN: ii libcudnn9
 TensorRT: 10.3.0.30
 Vision Works: NOT_INSTALLED
 VPI: 3.2.4
 Vulcan: 1.3.204


Installing Pytorch (venv): 

sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev
> system requirement 	

pip install --index-url https://developer.download.nvidia.com/compute/redist/jp/v61 torch --extra-index-url https://pypi.org/simple --dry-run
> Searches NVIDIA’s repository first (jp/v61) for JetPack-specific PyTorch.
> Falls back to PyPI (pypi.org/simple) for missing dependencies like filelock, typing_extensions, networkx, etc.

pip install torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v61 --extra-index-url https://pypi.org/simple


> damn. this ended up not being installed with cuda. it's torch cpu only... :(


wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
python3 -m pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

> compatibility issues between PyTorch and NumPy 2.0

(venv) machvision@machvision-desktop:~/Documents/senior-design/src/ml/object_detection/run$ pip uninstall torch torchvision numpy
(venv) machvision@machvision-desktop:~/Documents/senior-design/src/ml/object_detection/run$ pip install numpy==1.26.4
(venv) machvision@machvision-desktop:~/Documents/senior-design/src/ml/object_detection/run$ wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
(venv) machvision@machvision-desktop:~/Documents/senior-design/src/ml/object_detection/run$ pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
> torch works now!

sudo apt-get update
sudo apt-get install -y python3-pip libjpeg-dev libpng-dev libtiff-dev
> for torchvision

git clone --branch release/0.18 https://github.com/pytorch/vision torchvision
cd torchvision
python setup.py install
> this works 
