
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "device_picker.hpp"
#include "util.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <bits/stdc++.h>


#include <fstream>
#include <iostream>

#include "util.h"

std::string kernelsource =
    "__kernel void sigmoid(const int N, __global float* A, __global float* B){\n"
    "    int i = get_global_id(0);\n"
    "    int j = get_global_id(1);\n"
    "    int k = get_global_id(2);\n"
    "\n"
    "    if(i<N && j<N && k<N){             \n"
    "        for(int l=0; l<N; l++)         \n"
    "         {                             \n"
    "            int index = i*N*N*N + j*N*N + k*N + l;       \n"
    "            B[index] = 1/(1+exp(A[index]*-1));\n"
    "         }\n"
    "    }\n"
    "}\n";

#define TOL (0.001)
#define ORDER 2
int main(int argc, char *argv[]) {
  int N;

  long size;

  N = ORDER;
  size = N * N * N * N;

  double start_time; // Starting time
  double run_time;   // Timing
  util::Timer timer; // Timing

  std::vector<float> h_a(size); // Host memory for Matrix A
  std::vector<float> h_b(size); // Host memory for Matrix A

  cl::Buffer d_a, d_b; // Matrices in device memory

  try {

    cl_uint deviceIndex = 0;
    parseArguments(argc, argv, &deviceIndex);

    // Get list of devices
    std::vector<cl::Device> devices;
    unsigned numDevices = getDeviceList(devices);

    // Check device index in range
    if (deviceIndex >= numDevices) {
      std::cout << "Invalid device index (try '--list')\n";
      return EXIT_FAILURE;
    }

    cl::Device device = devices[deviceIndex];

    std::string name;
    getDeviceName(device, name);
    std::cout << "\nUsing OpenCL device: " << name << "\n";

    std::vector<cl::Device> chosen_device;
    chosen_device.push_back(device);

    cl::Context context(chosen_device);

    cl::Program program(context, kernelsource, true);

    cl::CommandQueue queue(context, device);

    init4D_vector(N, h_a, h_b);

    d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);

    d_b = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

    cl::make_kernel<int, cl::Buffer, cl::Buffer> sigmoidFunc(program, "sigmoid");

    {

      start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

      cl::NDRange global(N, N, N);

      sigmoidFunc(cl::EnqueueArgs(queue, global), N, d_a, d_b);

      queue.finish();

      run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) -
                 start_time;

      cl::copy(queue, d_b, h_b.begin(), h_b.end());

      int correct = 0;
      float tmp;
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          for (int k = 0; k < N; k++) {
            for (int l = 0; l < N; l++) {
              float aval = h_a[i * N * N * N + j * N * N + k * N + l];
              float bval = h_b[i * N * N * N + j * N * N + k * N + l];
              tmp = 1/(1+exp(aval*-1));
              tmp -= bval;

              if (tmp * tmp <
                  TOL * TOL) { // correct if square deviation is less
                correct++;     //  than tolerance squared
                printf("sigmoid of %.4f = %.4f   ", aval, bval);
              } else {
                printf(" tmp %f h_a %f h_b %f\n", tmp, aval, bval);
              }
            }
            printf("\n");
          }
        }
      }
    }

  } catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")"
              << std::endl;
  }

  return 0;
}