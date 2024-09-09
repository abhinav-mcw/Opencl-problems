ifndef CPPC
	CPPC=g++
endif

CPP_COMMON = ../Cpp_common

OPENCV_LIBPATH=/usr/include/opencv4
OPENCV_INCLUDEPATH=/usr/include/opencv4
# -lopencv_core -lopencv_imgproc -lopencv_highgui
OPENCV_LIBS=`pkg-config --cflags --libs opencv4`

CCFLAGS=

LIBS = -lOpenCL -fopenmp

INC = -I $(CPP_COMMON) -I ${OPENCV_LIBPATH} -I ${OPENCV_INCLUDEPATH}

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	LIBS = -framework OpenCL
endif

adaptive_mean: 
	${CPPC} adaptive_mean.cpp ${INC} ${CCFLAGS} ${LIBS}  $(OPENCV_LIBS) -o $@

clean:
	rm -f adaptive_mean