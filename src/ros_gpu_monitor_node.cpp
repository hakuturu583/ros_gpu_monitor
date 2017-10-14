#include <ros/ros.h>
#include <gpu_monitor.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "ros_gpu_monitor_node");
  gpu_monitor monitor;
  monitor.run();
  //ros::spin();
  return 0;
}
