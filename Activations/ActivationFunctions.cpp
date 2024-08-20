
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "device_picker.hpp"
#include "util.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>

#include "util.h"

#define TOL (0.001)
#define ORDER 2

float sigmoid(float x){
  return (float)(1/(1+exp(x*-1)));
}

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

    cl::Program program(context, util::loadProgram("./Relu/relu.cl"), true);

    cl::CommandQueue queue(context, device);

    init4D_vector(N, h_a, h_b);

    d_a = cl::Buffer(context, h_a.begin(), h_a.end(), true);

    d_b = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

    cl::make_kernel<int, cl::Buffer, cl::Buffer> reluFunc(program, "relu");

    {

      start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

      cl::NDRange global(N, N, N);

      reluFunc(cl::EnqueueArgs(queue, global), N, d_a, d_b);

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
              tmp = std::max(0.0f, aval);
              tmp -= bval;

              if (tmp * tmp <
                  TOL * TOL) { // correct if square deviation is less
                correct++;     //  than tolerance squared
                printf("relu of %.4f = %.4f   ", aval, bval);
              } else {
                printf(" tmp %f h_a %f h_b %f\n", tmp, aval, bval);
              }
            }
            printf("\n");
          }
        }
        
      }

      printf("%.2f seconds taken \n", run_time);
    }

    program = cl::Program(context, util::loadProgram("./Sigmoid/sigmoid.cl"), true);

    init4D_vector(N, h_a, h_b);

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
              tmp = sigmoid(aval);
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

    program = cl::Program(context, util::loadProgram("./Relu6/relu6.cl"), true);

    init4D_vector(N, h_a, h_b);

    cl::make_kernel<int, cl::Buffer, cl::Buffer> relu6Func(program, "relu6");

    {

      start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

      cl::NDRange global(N, N, N);

      relu6Func(cl::EnqueueArgs(queue, global), N, d_a, d_b);

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
              tmp = std::min(std::max(0.0f, aval), 6.0f);
              tmp -= bval;

              if (tmp * tmp <
                  TOL * TOL) { // correct if square deviation is less
                correct++;     //  than tolerance squared
                printf("relu6 of %.4f = %.4f   ", aval, bval);
              } else {
                printf(" tmp %f h_a %f h_b %f\n", tmp, aval, bval);
              }
            }
            printf("\n");
          }
        }
      }
    }

    program = cl::Program(context, util::loadProgram("./Tanh/tanh.cl"), true);

    init4D_vector(N, h_a, h_b);

    cl::make_kernel<int, cl::Buffer, cl::Buffer> tanhFunc(program, "tanhKernel");

    {

      start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

      cl::NDRange global(N, N, N);

      tanhFunc(cl::EnqueueArgs(queue, global), N, d_a, d_b);

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
              tmp = tanh(aval);
              tmp -= bval;

              if (tmp * tmp <
                  TOL * TOL) { // correct if square deviation is less
                correct++;     //  than tolerance squared
                printf("tanh of %.4f = %.4f   ", aval, bval);
              } else {
                printf(" tmp %f h_a %f h_b %f\n", tmp, aval, bval);
              }
            }
            printf("\n");
          }
        }
      }
    }

    program = cl::Program(context, util::loadProgram("./Gelu/gelu.cl"), true);

    init4D_vector(N, h_a, h_b);

    cl::make_kernel<int, cl::Buffer, cl::Buffer> geluFunc(program, "gelu");

    {

      start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

      cl::NDRange global(N, N, N);

      geluFunc(cl::EnqueueArgs(queue, global), N, d_a, d_b);

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
              tmp = .5f * aval * (1 + tanh(sqrt(2/3.14f) * (aval * 0.044715 * pow(aval,3))));
              tmp -= bval;

              if (tmp * tmp <
                  TOL * TOL) { // correct if square deviation is less
                correct++;     //  than tolerance squared
                printf("gelu of %.4f = %.4f   ", aval, bval);
              } else {
                printf(" tmp %f h_a %f h_b %f\n", tmp, aval, bval);
              }
            }
            printf("\n");
          }
        }
      }
    }

    program = cl::Program(context, util::loadProgram("./Silu/silu.cl"), true);

    init4D_vector(N, h_a, h_b);

    cl::make_kernel<int, cl::Buffer, cl::Buffer> siluFunc(program, "silu");

    {

      start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

      cl::NDRange global(N, N, N);

      siluFunc(cl::EnqueueArgs(queue, global), N, d_a, d_b);

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
              tmp = aval * sigmoid(aval);
              tmp -= bval;

              if (tmp * tmp <
                  TOL * TOL) { // correct if square deviation is less
                correct++;     //  than tolerance squared
                printf("silu of %.4f = %.4f   ", aval, bval);
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