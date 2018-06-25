CAFFE_LIB_DIR = /disk2/caffe_workspace/caffe-ssd/build/lib
CAFFE_INCLUDE_DIR = /disk2/caffe_workspace/caffe-ssd/include
PROJECT_DIR = ../..

all:    caffe_load_params  
#-fPIC
FLAGS +=   -DCPU_ONLY -DCAFFE_VERSION=1.0.0-rc3 -DNDEBUG -O2 -DUSE_OPENCV -DUSE_LEVELDB -DUSE_LMDB  -lpthread -DPARSE_PROTO_TREE

					
INCS_FLAGS = -I/usr/include/python2.7 \
			-I/usr/lib/python2.7/dist-packages/numpy/core/include \
			-I/usr/local/include \
			-I/usr/include/hdf5/serial  \
			-I/usr/local/lib/python2.7/dist-packages/numpy/core/include \
			-I$(CAFFE_INCLUDE_DIR) \
				

LIBS = -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lboost_regex -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb -lcaffe -lboost_thread -lstdc++ -lcblas -latlas -lblas -Wl,-rpath,\$ORIGIN/../../lib `pkg-config --cflags --libs opencv` 
		
LIBRARY_DIRS = -L/usr/local/lib \
				-L/usr/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
				-L$(CAFFE_LIB_DIR)
                

caffe_load_params:caffe_load_params.cpp  
	gcc -o $@ $^ $(INCS_FLAGS)  `pkg-config --cflags --libs opencv`  $(FLAGS) $(LIBRARY_DIRS) $(LIBS)
    
clean:
	rm -rf caffe_load_params  
