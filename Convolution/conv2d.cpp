#define __CL_ENABLE_EXCEPTIONS


#include "cl.hpp"
#include "device_picker.hpp"
#include "util.hpp"

#include <iostream>
#include <vector>

int main() {
    // 2D array dimensions
    const int N = 4;
    const int M = 2;

    int outputWidth = N - M + 1;

    // Input and output 2D arrays
    std::vector<float> input = {10, 25, 1, 5,    20, 32, 52, 1,    6, 4, 9, 40,    16, 13, 17,    29};
    std::vector<float> filter(M * M);
    std::vector<float> output(outputWidth * outputWidth, 0);

    // Initialize the input array with some values


    for (int y = 0; y < M; ++y) {
        float number ;
        for (int x = 0; x < M; ++x) {
            if(x==0)
                number = 1.0f;
            else
                number = 0.0f;
            filter[y * M + x] = number;
        }
    }

    try {
        // Get platforms and devices
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms.front();

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        cl::Device device = devices.front();

        // Create a context and command queue
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Build the program
        // program.build("-cl-std=CL1.2");
        cl::Program program(context, util::loadProgram("./conv2d.cl"), true);

        // Create buffers
        cl::Buffer buffer_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
        cl::Buffer buffer_filter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * filter.size(), filter.data());
        cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

        // Create the kernel and set arguments
        cl::Kernel kernel(program, "conv2d");
        kernel.setArg(0, buffer_input);
        kernel.setArg(1, buffer_filter);
        kernel.setArg(2, buffer_output);
        kernel.setArg(3, N);
        kernel.setArg(4, M);
        kernel.setArg(5, outputWidth);

        // Define the global work size (2D)
        cl::NDRange global_work_size(N, N);

        // Enqueue the kernel for execution
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, cl::NullRange);

        // Read the output buffer back to the host
        queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(float) * output.size(), output.data());

        // Output the result
        std::cout << "Output array:" << std::endl;
        for (int y = 0; y < outputWidth; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                std::cout << output[y * outputWidth + x] << " ";
            }
            std::cout << std::endl;
        }
    } catch (cl::Error &err) {
        std::cerr << "OpenCL error: " << err.what() << " (" << err.err() << ")" << std::endl;
    }

    return 0;
}