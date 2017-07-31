/* 
 * Copyright (C) 2017 daniele de gregorio, University of Bologna - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GNU GPLv3 license.
 *
 * please write to: d.degregorio@unibo.it
 */

#ifndef DARKNET_HPP
#define DARKNET_HPP

#include <string>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

//OPENCV
#include <opencv2/opencv.hpp>

//DARKNET
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "blas.h"
#include "image.h"
#include "layer.h"

//Darknet wrapper
#include <darknet_wrapper/DarkNetPrediction.hpp>
#include <darknet_wrapper/DarkNetUtilities.hpp>

namespace darknet_wrapper
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//// DarkNet configuration model
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

struct DarkNetConfiguration
{
    bool valid;
    std::string network_configuration_file;
    std::string network_weights_file;
    std::string output_names_file;

    float score_threshold;
    float threshold_hier;
    float nms;

    DarkNetConfiguration(std::string configuration_file = "")
    {
        if (!boost::filesystem::exists(configuration_file))
        {
            this->valid = false;
            return;
        }

        namespace pt = boost::property_tree;
        pt::ptree root;
        pt::read_json(configuration_file, root);
        network_configuration_file = root.get<std::string>("configuration_file");
        network_weights_file = root.get<std::string>("weights");
        output_names_file = root.get<std::string>("names");
        score_threshold = root.get<float>("score_threshold", 0.5);
        threshold_hier = root.get<float>("threshold_hier", 0.5);
        nms = root.get<float>("nms", .3);

        bool are_paths_relative = root.get<bool>("relative_paths");

        if (are_paths_relative)
        {
            boost::filesystem::path parent = boost::filesystem::path(configuration_file).parent_path().string();
            network_configuration_file = boost::filesystem::path(parent / network_configuration_file).string();
            network_weights_file = boost::filesystem::path(parent / network_weights_file).string();
            output_names_file = boost::filesystem::path(parent / output_names_file).string();
        }

        printf("DarkNetConfiguration:\n");
        printf(" - Network Configuration: %s\n", network_configuration_file.c_str());
        printf(" - Weights: %s\n", network_weights_file.c_str());
        printf(" - Names: %s\n", output_names_file.c_str());

        this->valid = true;
        this->valid &= boost::filesystem::exists(network_configuration_file);
        this->valid &= boost::filesystem::exists(network_weights_file);
        this->valid &= boost::filesystem::exists(output_names_file);
    }

    inline bool isValid()
    {
        return this->valid;
    }
};

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//// DarkNet Api Wrapper
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

class DarkNet
{
  private:
    DarkNetConfiguration _cfg;
    network _net;
    char **_output_names;

  public:
    DarkNet(std::string configuration_file)
    {
        this->_cfg = DarkNetConfiguration(configuration_file);
        if (!this->_cfg.isValid())
        {
            printf("DarkNet error: Invalid configuration!\n");
        }

        this->_output_names = get_labels((char *)this->_cfg.output_names_file.c_str());
        this->_net = parse_network_cfg((char *)this->_cfg.network_configuration_file.c_str());

        layer l = this->_net.layers[this->_net.n - 1];
        if (l.type == REGION)
        {
            printf("REGION %d", l.type);
        }

        printf("Setup: net.n = %d\n", this->_net.n);
        printf("net.layers[0].batch = %d\n", this->_net.layers[0].batch);

        load_weights(&(this->_net), (char *)this->_cfg.network_weights_file.c_str());
        printf("DarkNet Ready!");
    }

    DarkNetPredictionOutput predict(cv::Mat &img, float threshold = -1)
    {

        if (threshold < 0)
        {
            threshold = _cfg.score_threshold;
        }

        network &net = this->_net;

        //Creates ipl image of source
        cv::Mat target_img = img.clone();
        cv::cvtColor(target_img, target_img, CV_BGR2RGB);
        IplImage *iplImage;
        iplImage = cvCreateImage(cvSize(target_img.cols, target_img.rows), IPL_DEPTH_8U, 3);
        iplImage->imageData = (char *)target_img.data;

        image im = DarkNetUtilities::ipl_to_image(iplImage);
        image sized = letterbox_image(im, net.w, net.h);

        //Last layer dimension
        layer l = net.layers[net.n - 1];
        box *boxes = (box *)calloc(l.w * l.h * l.n, sizeof(box));
        float **probs = (float **)calloc(l.w * l.h * l.n, sizeof(float *));
        for (int j = 0; j < l.w * l.h * l.n; ++j)
            probs[j] = (float *)calloc(l.classes + 1, sizeof(float));

        //Prediction
        float *X = sized.data;
        network_predict(net, X);

        //Get output regions
        get_region_boxes(l, 1, 1, threshold, probs, boxes, 0, 0, _cfg.threshold_hier);

        //Refinement
        if (l.softmax_tree && _cfg.nms)
            do_nms_obj(boxes, probs, l.w * l.h * l.n, l.classes, _cfg.nms);
        else if (_cfg.nms)
            do_nms_sort(boxes, probs, l.w * l.h * l.n, l.classes, _cfg.nms);

        int prediction_number = l.w * l.h * l.n;
        float w_ration = float(target_img.cols) / float(sized.w);
        float h_ration = float(target_img.rows) / float(sized.h);

        //Builds Prediction Output
        DarkNetPredictionOutput prediction_output;

        for (int i = 0; i < prediction_number; ++i)
        {
            int class1 = max_index(probs[i], l.classes);

            float prob = probs[i][class1];
            if (prob > threshold)
            {
                //Wrapp Prediction in a user-friendly model, RESCALING frames to input images!
                DarkNetPrediction prediction(boxes[i], class1, prob, std::string(_output_names[class1]));
                prediction.rescale(im, net.w, net.h);
                prediction_output.predictions.push_back(prediction);
            }
        }

        //Free
        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w * l.h * l.n);
        return prediction_output;
    }
};
}
#endif /* DARKNET_HPP */
