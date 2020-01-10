nvcc = $(CUDA_ROOT)/bin/nvcc
cudalib = $(CUDA_ROOT)/lib64/
tensorflow = $(HOME)/anaconda3/envs/PPPU/lib/python3.5/site-packages/tensorflow/include
tf_lib = $(HOME)/anaconda3/envs/PPPU/lib/python3.5/site-packages/tensorflow

all: tf_nndistance_so.so
.PHONY : all

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -ltensorflow_framework -L $(tf_lib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

# render_balls_so.so: render_balls_so.cpp
# 	g++ -std=c++11 render_balls_so.cpp -o render_balls_so.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0


