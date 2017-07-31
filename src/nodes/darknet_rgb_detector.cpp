/*
   RGB_SCAN
   Developer: Daniele De Gregorio
 */

#include <darknet_wrapper/DarkNet.hpp>

//ROS
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <compressed_image_transport/compressed_subscriber.h>

//OPENCV
#include <opencv2/highgui/highgui.hpp>

//Ros
ros::NodeHandle *nh;

//Darknet
darknet_wrapper::DarkNet *darknet;

///////////////////////////////////////////////////////////
//// Does prediction on new Image
///////////////////////////////////////////////////////////

void manageNewImage(cv::Mat &img)
{
    darknet_wrapper::DarkNetPredictionOutput prediction_output = darknet->predict(img);

    for (int i = 0; i < prediction_output.predictions.size(); i++)
    {
        prediction_output.predictions[i].draw(img);
    }
    cv::imshow("output", img);
    cv::waitKey(1);
}

///////////////////////////////////////////////////////////
//// Image callback
///////////////////////////////////////////////////////////

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
    manageNewImage(img);
}

///////////////////////////////////////////////////////////
//// Main
///////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{

    //Init
    ros::init(argc, argv, "darknet_rgb_detector");
    nh = new ros::NodeHandle("~");

    //Params
    std::string camera_rgb_topic = nh->param<std::string>("camera_rgb_topic", "/camera/rgb/image_raw");
    bool compressed = nh->param<bool>("compressed_image", true);

    //Images callbacks
    image_transport::ImageTransport it(*nh);
    image_transport::Subscriber sub = it.subscribe(camera_rgb_topic, 1, imageCallback, ros::VoidPtr(), image_transport::TransportHints(compressed ? "compressed" : "raw"));

    //Darknet Wrapper Configuration
    std::string darkent_configuration = nh->param<std::string>("darknet_configuration", "");
    darknet = new darknet_wrapper::DarkNet(darkent_configuration);

    while (ros::ok())
    {
        ros::spinOnce();
    }
}