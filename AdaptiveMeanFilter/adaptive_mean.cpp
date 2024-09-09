#define __CL_ENABLE_EXCEPTIONS

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "cl.hpp"
#include "util.hpp"
#include "err_code.h"
#include "device_picker.hpp"

cl::Image2D LoadImage(cl::Context context, char *fileName, int &width, int &height)
{
    cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);
    cv::Mat imageRGBA;
    
    width = image.cols;
    height = image.rows;
    
    cv::cvtColor(image, imageRGBA, cv::COLOR_RGB2RGBA);

    char *buffer = reinterpret_cast<char *>(imageRGBA.data);
    
    cl::Image2D clImage(context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
                            width,
                            height,
                            0,
                            buffer);
    return clImage;
}

int main(int argc, char *argv[])
{

    char * filename = "images/sunset.jpg";
    char * filename_out = "images/sunset_mean_filter.jpg";

    try{

    //     std::vector<cl::Platform> platforms;
    // cl::Platform::get(&platforms);
    // std::cout << "\nNumber of OpenCL plaforms: " << platforms.size() << std::endl;

        cl_uint deviceIndex = 0;
        parseArguments(argc, argv, &deviceIndex);

        // // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // // // Check device index in range
        // if (deviceIndex >= numDevices)
        // {
        //   std::cout << "Invalid device index (try '--list')\n";
        //   return EXIT_FAILURE;
        // }

        // cl::Device device = devices[deviceIndex];

        // std::string name;
        // getDeviceName(device, name);
        // std::cout << "\nUsing OpenCL device: " << name << "\n";

        // std::vector<cl::Device> chosen_device;
        // chosen_device.push_back(device);

        // cl::Context context(chosen_device);

        // cl::CommandQueue queue(context, device);

        // //Load input image to the host and to the CPU
        // int width, height;
        // int maskSize = 5;

        // cl::Image2D clImageInput;
        // clImageInput = LoadImage(context, filename, width, height);
        // printf("\nwidth: %d height: %d \n",width,height);
        
        //Create output image object
        // cl::Image2D imageOutput(context,
        //             CL_MEM_WRITE_ONLY,
        //             cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
        //             width,
        //             height,
        //             0,
        //             NULL);
        
        // cl::Program program(context, util::loadProgram("adaptive_mean.cl"), true);

        // cl::make_kernel<cl::Image2D, cl::Image2D, int, int, int> adaptiveMean(program, "AdaptiveMean");

        // cl::NDRange global(width, height);

        // adaptiveMean(cl::EnqueueArgs(queue, global), clImageInput, imageOutput, maskSize, width, height);

        // queue.finish();

        // cl_uint8* oup = new cl_uint8[width * height];

        // cl::size_t<3> origin;
        // origin[0] = 0; origin[1] = 0, origin[2] = 0;
        // cl::size_t<3> region;
        // region[0] = width; region[1] = height; region[2] = 1;

        // queue.enqueueReadImage(imageOutput, CL_TRUE, origin, region, 0, 0, oup);

        // cv::imwrite(filename_out,  cv::Mat(height, width, CV_8UC4, oup));

    } catch(cl::Error err){
        std::cout << "Exception\n";
        std::cout << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
    return 0;
}