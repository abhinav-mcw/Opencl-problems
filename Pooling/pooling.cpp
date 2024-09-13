#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "device_picker.hpp"
#include "util.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

int main()
{
    // 2D array dimensions
    const int N = 4;
    const int F = 2;

    const int S = 2;

    int outputWidth = floor((N - F) / S + 1);

    // Input and output 2D arrays
    std::vector<float> input = {1, 3, 2, 1, 2, 9, 1, 1, 1, 4, 2, 3, 5, 6, 1, 2};
    std::vector<float> output_max(outputWidth * outputWidth, 0);
    std::vector<float> output_avg(outputWidth * outputWidth, 0);

    try
    {
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
        cl::Program program(context, util::loadProgram("./pooling.cl"), true);

        // // Create buffers
        cl::Buffer buffer_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());

        cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * output_max.size());

        // // Define the global work size (2D)
        cl::NDRange global_work_size(N, N);

        // // Create the kernel and set arguments
        cl::Kernel kernel_max(program, "max_pooling");
        kernel_max.setArg(0, buffer_input);
        kernel_max.setArg(1, buffer_output);
        kernel_max.setArg(2, N);
        kernel_max.setArg(3, outputWidth);
        kernel_max.setArg(4, F);
        kernel_max.setArg(5, S);


        // // Enqueue the kernel for execution
        queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, global_work_size, cl::NullRange);

        // // Read the output buffer back to the host
        queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(float) * output_max.size(), output_max.data());

        // // Output the result
        std::cout << "Output pool max array:" << std::endl;
        for (int y = 0; y < outputWidth; ++y)
        {
            for (int x = 0; x < outputWidth; ++x)
            {
                std::cout << output_max[y * outputWidth + x] << " ";
            }
            std::cout << std::endl;
        }

        cl::Buffer buffer_output_avg(context, CL_MEM_WRITE_ONLY, sizeof(float) * output_avg.size());

        // Create the kernel for avg_pooling
        cl::Kernel kernel_avg(program, "avg_pooling");
        kernel_avg.setArg(0, buffer_input);
        kernel_avg.setArg(1, buffer_output_avg);
        kernel_avg.setArg(2, N);
        kernel_avg.setArg(3, outputWidth);
        kernel_avg.setArg(4, F);
        kernel_avg.setArg(5, S);


        // Enqueue the avg_pooling kernel for execution
        queue.enqueueNDRangeKernel(kernel_avg, cl::NullRange, global_work_size, cl::NullRange);

        // Read the output buffer back to the host for avg_pooling
        queue.enqueueReadBuffer(buffer_output_avg, CL_TRUE, 0, sizeof(float) * output_avg.size(), output_avg.data());

        std::cout << "Output pool avg array:" << std::endl;
        for (int y = 0; y < outputWidth; ++y)
        {
            for (int x = 0; x < outputWidth; ++x)
            {
                std::cout << output_avg[y * outputWidth + x] << " ";
            }
            std::cout << std::endl;
        }
    }
    catch (cl::Error &err)
    {
        std::cerr << "OpenCL error: " << err.what() << " (" << err.err() << ")" << std::endl;
    }

    return 0;
}