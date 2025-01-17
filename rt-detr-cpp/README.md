# RT-DETR LibTorch Inference in C++

Download LibTorch:

```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/appuser/object_detection/rt-detr-cpp/libtorch/lib:$LD_LIBRARY_PATH

```

Build the project:

```bash
mkdir build
cd build
cmake ..
make
```
