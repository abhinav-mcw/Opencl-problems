
ifndef CPPC
	CPPC=g++
endif

CPP_COMMON = ../Cpp_common

UTIL = ./Util

CCFLAGS=

INC1 = -I $(CPP_COMMON) 
INC2 = -I $(UTIL)

LIBS = -lOpenCL

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	LIBS = -framework OpenCL
endif

ActivationFunctions: ActivationFunctions.cpp
	$(CPPC) $^ $(INC1) ${INC2} $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f ActivationFunctions
