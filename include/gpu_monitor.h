#ifndef GPU_MONITOR_INCLUDED
#define GPU_MONITOR_INCLUDED

//headers in this package
#include <ros_gpu_monitor/GpuProperties.h>

//headers for ROS
#include <ros/ros.h>

//headers for cuda
#include <cuda_runtime.h>
#include <cuda.h>

class gpu_monitor
{
public:
  gpu_monitor();
  ~gpu_monitor();
  void run();
private:
  ros::NodeHandle nh_;
  bool get_device_info_(int device_id);
  cudaDeviceProp device_properties_;
  ros_gpu_monitor::GpuProperties gpu_properties_msg_;
  ros::Publisher gpu_properties_pub;
  int publish_rate_;
};
#endif
