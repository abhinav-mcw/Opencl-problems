
ifndef CPPC
	CPPC=g++
endif

CPP_COMMON = ../Cpp_common

CCFLAGS=-std=c++14 -DWITH_OPENCL

INC = -I $(CPP_COMMON) -I /usr/include/CL/

LIBS_CL = -lOpenCL -lrt 
LIBS_CV = `pkg-config --cflags --libs opencv4`

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -stdlib=libc++
	LIBS = -framework OpenCL
endif

CCFLAGS += -D DEVICE=$(DEVICE)

mirror_horizontal: mirror_horizontal.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS_CV) $(LIBS_CL) -o $@


clean:
	rm -f mirror_horizontal