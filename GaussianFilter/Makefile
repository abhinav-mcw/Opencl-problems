ifndef CPPC
	CPPC=g++
endif

CPP_COMMON = ../Cpp_common

OPENCV_LIBPATH=/usr/include/opencv4
OPENCV_INCLUDEPATH=/usr/include/opencv4
# -lopencv_core -lopencv_imgproc -lopencv_highgui
OPENCV_LIBS=`pkg-config --cflags --libs opencv4`

CCFLAGS=-O3 -ffast-math

LIBS = -lm -lOpenCL -fopenmp

INC = -I $(CPP_COMMON) -I $(OPENCV_LIBPATH) -I $(OPENCV_INCLUDEPATH)

GAUSS_OBJS = gauss.o matrix_lib.o
EXEC = gauss

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -stdlib=libc++
	LIBS = -lm -framework OpenCL
endif

all: $(EXEC)

gauss: $(GAUSS_OBJS)
	$(CPPC) $(GAUSS_OBJS) $(CCFLAGS) -L $(OPENCV_LIBPATH) $(LIBS) $(OPENCV_LIBS) -o $(EXEC) $(INC)

.c.o:
	$(CPPC) -c $< $(CCFLAGS) -o $@

.cpp.o:
	$(CPPC) -c $< $(CCFLAGS) $(INC) -o $@

gauss.o:	gauss.hpp matrix_lib.hpp

matrix_lib.o:	gauss.hpp

clean:
	rm -f $(GAUSS_OBJS) $(EXEC)