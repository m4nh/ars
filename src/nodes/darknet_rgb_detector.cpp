/*
   RGB_SCAN
   Developer: Daniele De Gregorio
 */

#include <darknet_wrapper/DarkNet.hpp>

//ROS
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <compressed_image_transport/compressed_subscriber.h>

//OPENCV
#include <opencv2/highgui/highgui.hpp>

#define MAX_PREDICTIONS 32

//Ros
ros::NodeHandle *nh;
ros::Publisher predictions_publisher;
bool publish_output;

//Darknet
darknet_wrapper::DarkNet *darknet;

///////////////////////////////////////////////////////////
//// Does prediction on new Image
///////////////////////////////////////////////////////////

darknet_wrapper::DarkNetPredictionOutput manageNewImage(cv::Mat &img)
{
    darknet_wrapper::DarkNetPredictionOutput prediction_output = darknet->predict(img);

    for (int i = 0; i < prediction_output.predictions.size(); i++)
    {
        prediction_output.predictions[i].draw(img);
    }
    cv::imshow("output", img);
    cv::waitKey(1);

    return prediction_output;
}

///////////////////////////////////////////////////////////
//// Image callback
///////////////////////////////////////////////////////////

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
    darknet_wrapper::DarkNetPredictionOutput prediction_output = manageNewImage(img);

    if (publish_output)
    {
        int predictions_number = int(prediction_output.predictions.size());

        std_msgs::Float64MultiArray prediction_msg;
        prediction_msg.data.resize(1 + MAX_PREDICTIONS * 5);
        prediction_msg.data[0] = predictions_number;
        for (int i = 0; i < predictions_number; i++)
        {
            prediction_msg.data[1 + i * 5] = prediction_output.predictions[i].prediction_box.x;
            prediction_msg.data[1 + i * 5 + 1] = prediction_output.predictions[i].prediction_box.y;
            prediction_msg.data[1 + i * 5 + 2] = prediction_output.predictions[i].prediction_box.w;
            prediction_msg.data[1 + i * 5 + 3] = prediction_output.predictions[i].prediction_box.h;
            prediction_msg.data[1 + i * 5 + 4] = prediction_output.predictions[i].prediction_class;
        }

        predictions_publisher.publish(prediction_msg);
    }
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

    //Output Configuration
    std::string output_topic_name = nh->param<std::string>("output_topic_name", "predictions");
    publish_output = nh->param<bool>("publish_output", true);
    predictions_publisher = nh->advertise<std_msgs::Float64MultiArray>(output_topic_name, 1);

    while (ros::ok())
    {
        ros::spinOnce();
    }
}