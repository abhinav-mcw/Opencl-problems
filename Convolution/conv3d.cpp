#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "device_picker.hpp"
#include "util.hpp"

#include <iostream>
#include <vector>

int main()
{
	// 3D array dimensions
	const int N = 4;
	const int M = 2;

	int outputDim = N - M + 1;

	std::vector<float> input = {1, 0, 1, 0, 1, 1, 3, 1, 1, 1, 0, 2, 0, 2, 1, 1,
								1, 0, 0, 1, 2, 0, 1, 2, 3, 1, 1, 1, 0, 0, 3, 1,
								2, 0, 1, 1, 3, 3, 1, 0, 2, 1, 1, 0, 3, 2, 1, 2,
								1, 0, 2, 0, 1, 0, 3, 3, 3, 1, 0, 0, 1, 1, 0, 2};

	std::vector<float> filter = {0, 1, 0, 0, 2, 1, 0, 0};

	std::vector<float> output(outputDim * outputDim * outputDim, 0);

	try
	{
		// Get platforms and devices
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		cl::Platform platform = platforms.front();

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		cl::Device device = devices.front();

		// // Create a context and command queue
		cl::Context context(device);
		cl::CommandQueue queue(context, device);      

		cl::Program program(context, util::loadProgram("./conv3d.cl"), true);

		cl::Buffer buffer_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
        cl::Buffer buffer_filter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * filter.size(), filter.data());
        cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

		cl::Kernel kernel(program, "conv3d");
        kernel.setArg(0, buffer_input);
        kernel.setArg(1, buffer_filter);
        kernel.setArg(2, buffer_output);
        kernel.setArg(3, N);
        kernel.setArg(4, M);
        kernel.setArg(5, outputDim);

		cl::NDRange global_work_size(N, N, N);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, cl::NullRange);

		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data());

		std::cout << "Output array:" << std::endl;
        for(int z=0; z<outputDim; z++){
			for (int y = 0; y < outputDim; ++y) {
				for (int x = 0; x < outputDim; ++x) {
					std::cout << output[z*outputDim*outputDim + y*outputDim + x] << " ";
				}
            	std::cout << std::endl;
        	}
		}
	}
	catch (cl::Error &err)
	{
		std::cerr << "OpenCL error: " << err.what() << " (" << err.err() << ")"
				  << std::endl;
	}

	return 0;
}