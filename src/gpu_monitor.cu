#include <gpu_monitor.h>
#include <ros_gpu_monitor/GPUProperties.h>
#include <ros_gpu_monitor/CudaInfo.h>
#include <ros_gpu_monitor/BytesWithUnit.h>
#include <ros_gpu_monitor/HzWithUnit.h>

//headers for cuda
#include <cuda_runtime.h>
#include <cuda.h>

gpu_monitor::gpu_monitor()
{
  int device_id;
  nh_.param<int>("device_id", device_id, 0);
  nh_.param<int>("publish_rate", publish_rate_, 10);
  get_device_info_(device_id);
  gpu_properties_pub = nh_.advertise<ros_gpu_monitor::GpuProperties>("gpu_properties", 1);
}

gpu_monitor::~gpu_monitor()
{

}

void gpu_monitor::run()
{
  ros::Rate rate = ros::Rate(publish_rate_);
  while (ros::ok())
  {
    gpu_properties_pub.publish(gpu_properties_msg_);
    rate.sleep();
  }
}

void gpu_monitor::get_device_info_(int device_id)
{
  cudaSetDevice(device_id);
  cudaGetDeviceProperties(&device_properties_, device_id);
  ROS_INFO_STREAM("cuda device type = " << device_properties_.name);
  gpu_properties_msg_.gpu_type = device_properties_.name;
  gpu_properties_msg_.device_id = device_id;
  int runtime_version = 0;
  int driver_version = 0;
  cudaRuntimeGetVersion(&runtime_version);
  cudaDriverGetVersion(&driver_version);
  gpu_properties_msg_.cuda_info.runtime_version = (float)runtime_version/1000;
  gpu_properties_msg_.cuda_info.driver_version = (float)driver_version/1000;
  gpu_properties_msg_.global_memory.bytes = (float)device_properties_.totalGlobalMem/1048576.0f;
  gpu_properties_msg_.global_memory.data_unit = gpu_properties_msg_.global_memory.MB;
  gpu_properties_msg_.multiprocessors = device_properties_.multiProcessorCount;
  //gpu_properties_msg_.cuda_cores = _ConvertSMVer2Cores(device_properties_.major, device_properties_.minor) * device_properties_.multiProcessorCount);
  gpu_properties_msg_.gpu_max_clock_rate.hz = device_properties_.clockRate*1e-6f;
  gpu_properties_msg_.gpu_max_clock_rate.data_unit = gpu_properties_msg_.gpu_max_clock_rate.GHZ;
  gpu_properties_msg_.cuda_info.capability_major_version = device_properties_.major;
  gpu_properties_msg_.cuda_info.capability_minor_version = device_properties_.minor;
}
