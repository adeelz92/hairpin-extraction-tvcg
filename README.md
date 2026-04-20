# hairpin-flow

# [Datasets](https://uofh-my.sharepoint.com/:f:/g/personal/azafar3_cougarnet_uh_edu/IgBCUCBVRQBZRp51IhJ49ba2AT6kiBjjf2LKxVbrZ1IzXEY?e=QF69kG)

# Extract hairpin vortices
Extract hairpin vortices in turbulent flows based on geometric and physical analysis.

Installation instructions Windows 10.

Prerequisites: 
1. CMake >= 3.20. Get the installer from https://cmake.org/download/.
Make sure to select "Add CMake to the System Path for the Current User" 
when prompt.

2. Git. Get it from https://git-scm.com/downloads. Install with the default options. 

3. Visual Studio Community 2022. Get it from https://visualstudio.microsoft.com/vs/community/.
While installing, select the package "Desktop development with C++".

4. Download and install VTK.
	
	Open Command Prompt and cd to where you want to build and install VTK. 
	Execute the following commands.

	```
	git clone https://github.com/Kitware/VTK.git. 
	```

	This will download the VTK source code and make a folder VTK. Let Path/To/VTK is the VTK source directory.

	```
	cd VTK & git checkout v9.3.1
	```

	```
	cd .. & mkdir build & cd build
	```

	```
	cmake -DCMAKE_BUILD_TYPE=Release -DVTK_MODULE_ENABLE_VTK_FiltersParallelDIY2=YES ..
	```

	You should see "Configure Done" and "Generating Done" after the configuration is finished.
	
	Run Visual Studio as administrator. Select "Open Project or Solution" and 
	open the VTK solution file in Path/To/VTK/build folder.
	
	Change the build type from "Debug" to "Release" on top of the visual studio.
	
	Right click ALL_BUILD in the solution explorer and press Build. Wait for the build process to finish. 
	
	Make sure you get "========== Build: 259 succeeded, 0 failed, 0 up-to-date, 0 skipped ==========" at the end of the build process.
	
	Now in the Solution Explorer scroll down to INSTALL, right click and select Build. This will build the required headers and libraries in C:\Program Files (x86)\VTK or C:\Program Files\VTK.

	Add C:\Program Files (x86)\VTK\bin to the Path environment variable and restart the system.
			
5. Download and Configure CGAL. In command prompt, cd to the directory where you want to install Vcpkg.

	```
	git clone https://github.com/microsoft/vcpkg
	cd vcpkg & .\bootstrap-vcpkg.bat
	vcpkg.exe install yasm-tool:x86-windows
	vcpkg.exe install cgal:x64-windows eigen3:x64-windows
	```
	
6. Download and install TTK.
	
	1. Download https://github.com/topology-tool-kit/ttk/archive/1.3.0.zip
	2. Extract to ttk-1.3.0
	3. cd ttk-1.3.0 & mkdir build & cd build & cmake-gui ..
	4. Set TTK_BUILD_PARAVIEW_PLUGINS to OFF
	5. Configure & Generate
	6. Open visual studio as administrator and open ttk.sln.
	7. Change the build type to "Release".
	8. Right click "ALL_BUILD" and build.
	9. Right click "INSTALL" and build.
	10. Add "C:\Program Files\ttk\bin" to Path environment variable.
	
7. Build this project.

	Clone this repository using git or download the code directly.
	In Command Prompt, cd to Path/To/Code.

	```
	mkdir build & cd build & cmake-gui ..
	```
	
	Press Configure.

 	Select x64 in "Optional Platform for Generator".
	
	Select the option "Specify toolchain file for cross-compiling" and click Next. 
	
	Select the file Path/To/vcpkg/scripts/buildsystems/vcpkg.cmake and click Finish. 
	
	Press Configure and let the cmake configure files.
	
	After the configuration is done and you see the message "Configuring done", press Generate. 
	
	After "Generating done", select Open Project. Right click "ALL_BUILD" and build.
	
8. Run the code on a new dataset e.g.. In the Command Prompt, cd to Path/To/Code/build/Release and run the following commands in order.
	
	```
	Region_Growing.exe "Path\To\dataset.vtk" "lambda2 threshold"
	Contour_Splitting.exe "Path\To\dataset_Regions.vtk"
	Extract_Hairpins.exe "Path\To\dataset_Regions_Split.vtk"
	```
	
	Fix point ids using the fix_vortex_point_ids.py
	
	```
	Eval_Hairpins.exe "Path\To\dataset_GTs.vtk" "Path\To\dataset_Regions_Split_hairpinRegion_with_ids.vtk" 0.5
	```
	
	For the channel flow, e.g.
	
	```
	Region_Growing.exe "C:\flow_datasets\channel.vtk" "-13.395"
	Contour_Splitting.exe "C:\flow_datasets\channel_Regions.vtk"
	Extract_Hairpins.exe "C:\flow_datasets\channel_Regions_Split.vtk"
	```
	
	Fix point ids using the fix_vortex_point_ids.py code and change the paths in the code.
	
	```
	Eval_Hairpins.exe "C:\flow_datasets\channel_GTs.vtk" "C:\flow_datasets\channel_Regions_Split_hairpinRegion_with_ids.vtk" 0.5
	```

 # Linux Setup Guide

## 1. Install VTK

### i. If you have root access:

```bash
sudo yum update -y
sudo yum groupinstall "Development Tools" -y
sudo yum install cmake git gcc gcc-c++ python3-devel mesa-libGL-devel mesa-libGLU-devel libXt-devel -y
```

### ii. If you don't have root access (e.g., on HPC):

```bash
module load cmake gcc
```

### iii. Clone and build VTK:

```bash
git clone https://github.com/Kitware/VTK.git
cd VTK
git checkout v9.3.1
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/VTK/install -DVTK_MODULE_ENABLE_VTK_FiltersParallelDIY2=YES ..
make -j$(nproc)
make install
```

### iv. Export VTK environment variables:

```bash
export PATH=$HOME/VTK/install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/VTK/install/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$HOME/VTK/install:$CMAKE_PREFIX_PATH
```

---

## 2. Install CGAL

### i. Create directory and install dependencies

```bash
mkdir -p $HOME/tools/cgal_deps
cd $HOME/tools/cgal_deps
```

#### GMP

```bash
wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
tar -xf gmp-6.3.0.tar.xz
cd gmp-6.3.0
./configure --prefix=$HOME/tools/gmp
make -j$(nproc)
make install
cd ..
```

#### MPFR

```bash
wget https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.1.tar.gz
tar -xzf mpfr-4.2.1.tar.gz
cd mpfr-4.2.1
./configure --prefix=$HOME/tools/mpfr --with-gmp=$HOME/tools/gmp
make -j$(nproc)
make install
cd ..
```

#### Boost

```bash
wget https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
tar -xf boost-1.84.0.tar.xz
cd boost-1.84.0
./bootstrap.sh --prefix=$HOME/tools/boost
./b2 install
cd ..
```

#### Eigen

```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
mv eigen-3.4.0 $HOME/tools/eigen
cd ..
```

### ii. Install CGAL

```bash
wget https://github.com/CGAL/cgal/releases/download/v6.0/CGAL-6.0.tar.xz
tar -xf CGAL-6.0.tar.xz
cd CGAL-6.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/tools/cgal -DGMP_INCLUDE_DIR=$HOME/tools/gmp/include -DGMP_LIBRARIES=$HOME/tools/gmp/lib/libgmp.so -DMPFR_INCLUDE_DIR=$HOME/tools/mpfr/include -DMPFR_LIBRARIES=$HOME/tools/mpfr/lib/libmpfr.so -DBOOST_ROOT=$HOME/tools/boost -DEIGEN3_INCLUDE_DIR=$HOME/tools/eigen ..
make -j$(nproc)
make install
```

### iii. Export environment variables for CGAL

```bash
export PATH=$HOME/tools/cgal/bin:$PATH
export LD_LIBRARY_PATH=$HOME/tools/gmp/lib:$HOME/tools/mpfr/lib:$LD_LIBRARY_PATH

export BOOST_ROOT=$HOME/tools/boost

export GMP_DIR=$HOME/tools/gmp
export CMAKE_PREFIX_PATH=$GMP_DIR:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$GMP_DIR/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$GMP_DIR/lib/pkgconfig:$PKG_CONFIG_PATH

export MPFR_DIR=$HOME/tools/mpfr
export CMAKE_PREFIX_PATH=$MPFR_DIR:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$MPFR_DIR/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$MPFR_DIR/lib/pkgconfig:$PKG_CONFIG_PATH

export EIGEN_HOME=$HOME/tools/eigen
```

---

## 3. Install TTK

### i. Download and build TTK

```bash
cd $HOME/tools
wget https://github.com/topology-tool-kit/ttk/archive/refs/tags/1.3.0.zip
unzip 1.3.0.zip
cd ttk-1.3.0
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DTTK_BUILD_PARAVIEW_PLUGINS=OFF -DCMAKE_INSTALL_PREFIX=$HOME/tools/ttk-1.3.0/ttk-install ..
make -j$(nproc)
make install
```

### ii. Export environment variables for TTK

```bash
export PATH=$HOME/tools/ttk-1.3.0/ttk-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/tools/ttk-1.3.0/ttk-install/lib:$LD_LIBRARY_PATH
source ~/.bashrc
```

---

## 4. Run the Hairpin Flow Project

### i. Build the project

```bash
cd path/to/hairpin-flow-main
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### ii. Run the binaries

```
./Region_Growing "Path\To\dataset.vtk" "lambda2 threshold"
./Contour_Splitting "Path\To\dataset_Regions.vtk"
./Extract_Hairpins "Path\To\dataset_Regions_Split.vtk"
```

Fix point ids using the fix_vortex_point_ids.py

```
./Eval_Hairpins "Path\To\dataset_GTs.vtk" "Path\To\dataset_Regions_Split_hairpinRegion_with_ids.vtk" 0.5
```

For the channel flow, e.g.

```
./Region_Growing "C:\flow_datasets\channel.vtk" "-13.395"
./Contour_Splitting "C:\flow_datasets\channel_Regions.vtk"
./Extract_Hairpins "C:\flow_datasets\channel_Regions_Split.vtk"
```

Fix point ids using the fix_vortex_point_ids.py code and change the paths in the code.

```
./Eval_Hairpins "C:\flow_datasets\channel_GTs.vtk" "C:\flow_datasets\channel_Regions_Split_hairpinRegion_with_ids.vtk" 0.5
```