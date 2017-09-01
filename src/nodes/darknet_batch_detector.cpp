/*
   RGB_SCAN
   Developer: Daniele De Gregorio
 */

#include <fstream>

#include <darknet_wrapper/DarkNet.hpp>

//ROS
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <compressed_image_transport/compressed_subscriber.h>

//BOOST
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

//OPENCV
#include <opencv2/highgui/highgui.hpp>

#define MAX_PREDICTIONS 32

namespace fs = boost::filesystem;

//Ros
ros::NodeHandle *nh;
ros::Publisher predictions_publisher;
bool publish_output;

//Darknet
darknet_wrapper::DarkNet *darknet;

///////////////////////////////////////////////////////////
//// Main
///////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{

    //Init
    ros::init(argc, argv, "darknet_batch_detector");
    nh = new ros::NodeHandle("~");

    //Params
    std::string dataset_path_str = nh->param<std::string>("dataset_path", "");

    fs::path dataset_path(dataset_path_str);
    fs::path images_path = dataset_path / fs::path("images");
    fs::path detections_path = dataset_path / nh->param<std::string>("detections_subfolder", "yolo_detections");

    if (!boost::filesystem::exists(images_path))
    {
        ROS_INFO_STREAM("Path doesn't exist " << images_path);
        exit(0);
    }

    if (!boost::filesystem::exists(detections_path))
    {
        boost::filesystem::create_directory(detections_path);
    }

    //Darknet Wrapper Configuration
    std::string darkent_configuration = nh->param<std::string>("darknet_configuration", "");
    darknet = new darknet_wrapper::DarkNet(darkent_configuration);

    //Load images
    std::vector<std::string> images_names;
    std::vector<darknet_wrapper::DarkNetPredictionOutput> predictions;

    fs::directory_iterator it(images_path), eod;
    BOOST_FOREACH (fs::path const &p, std::make_pair(it, eod))
    {
        if (fs::is_regular_file(p))
        {
            std::string name = p.stem().string();
            cv::Mat img = cv::imread(p.string());
            images_names.push_back(name);

            darknet_wrapper::DarkNetPredictionOutput prediction_output = darknet->predict(img);
            predictions.push_back(prediction_output);

            fs::path detection_file_path = detections_path / std::string(name + ".txt");
            std::ofstream detection_file(detection_file_path.string());
            detection_file << prediction_output;
            detection_file.close();

            ROS_INFO_STREAM(name);
            ROS_INFO_STREAM(prediction_output);
        }
    }

    exit(0);
}