/* 
 * Copyright (C) 2017 daniele de gregorio, University of Bologna - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GNU GPLv3 license.
 *
 * please write to: d.degregorio@unibo.it
 */

#ifndef DARKNETPREDICTION_HPP
#define DARKNETPREDICTION_HPP

#include <string>
#include <fstream>
#include <sstream>
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
#include <darknet_wrapper/DarkNetUtilities.hpp>

namespace darknet_wrapper
{

class DarkNetPrediction
{
  public:
    box prediction_box;
    int prediction_class;
    float prediction_prob;
    std::string name;

    DarkNetPrediction(box prediction_box, int prediction_class, float prediction_prob = 0.0, std::string name = "")
    {
        this->prediction_box = prediction_box;
        this->prediction_class = prediction_class;
        this->prediction_prob = prediction_prob;
        this->name = name;
    }

    void rescale(image &image, int w, int h)
    {
        int new_w = image.w;
        int new_h = image.h;
        if (((float)w / image.w) < ((float)h / image.h))
        {
            new_w = w;
            new_h = (image.h * w) / image.w;
        }
        else
        {
            new_h = h;
            new_w = (image.w * h) / image.h;
        }

        float diffw = fabs(w - new_w) * 0.5f;
        float diffh = fabs(h - new_h) * 0.5f;

        float bx = this->prediction_box.x * w;
        float by = this->prediction_box.y * h;
        float bw = this->prediction_box.w * w;
        float bh = this->prediction_box.h * h;

        bx -= diffw;
        by -= diffh;

        this->prediction_box.x = bx / float(new_w);
        this->prediction_box.y = by / float(new_h);
        this->prediction_box.w = bw / float(new_w);
        this->prediction_box.h = bh / float(new_h);
    }

    static cv::Point3i getColorByIndex(int object_index)
    {
        if (object_index < 0)
            return cv::Point3i(255, 255, 255);

        //printf("REQUEST INDEX %d\n", object_index);
        if (object_index < 11)
        {
            if (object_index == 0)
                return cv::Point3i(244, 67, 54); //Boxgreen
            if (object_index == 1)
                return cv::Point3i(33, 150, 243); //Boxmouse
            if (object_index == 2)
                return cv::Point3i(76, 175, 80); //Boxred
            if (object_index == 3)
                return cv::Point3i(255, 193, 7); //Chamo
            if (object_index == 4)
                return cv::Point3i(156, 39, 176); //Cupgreen
            if (object_index == 5)
                return cv::Point3i(96, 125, 139); //Cupred
            if (object_index == 6)
                return cv::Point3i(255, 166, 0); //Detergent
            if (object_index == 7)
                return cv::Point3i(255, 217, 20); //Glue
            if (object_index == 8)
                return cv::Point3i(255, 107, 77); //Korn
            if (object_index == 9)
                return cv::Point3i(255, 255, 0); //Multimeter
            if (object_index == 10)
                return cv::Point3i(160, 206, 121); //Robot
            if (object_index == 11)
                return cv::Point3i(197, 184, 98); //Talc
            return cv::Point3i(255, 255, 255);
        }
        return cv::Point3i(0, 0, 0);
    }

    cv::Scalar getColor(int classes)
    {
        cv::Point3i color = DarkNetPrediction::getColorByIndex(this->prediction_class);
        return cv::Scalar(color.z, color.y, color.x);
    }

    void draw(cv::Mat &img)
    {
        box b = this->prediction_box;
        int left = (b.x - b.w / 2.) * img.cols;
        int right = (b.x + b.w / 2.) * img.cols;
        int top = (b.y - b.h / 2.) * img.rows;
        int bot = (b.y + b.h / 2.) * img.rows;

        if (left < 0)
            left = 0;
        if (right > img.cols - 1)
            right = img.cols - 1;
        if (top < 0)
            top = 0;
        if (bot > img.rows - 1)
            bot = img.rows - 1;

        cv::Scalar color = DarkNetUtilities::getColor(this->prediction_class);
        cv::rectangle(img, cv::Point(left, top), cv::Point(right, bot), color, 2);
        cv::rectangle(img, cv::Point(left, top - 30), cv::Point(right, top), color, 2);
        cv::rectangle(img, cv::Point(left, top - 30), cv::Point(right, top), color, -1);
        cv::putText(img, this->name, cv::Point(left, top), cv::FONT_HERSHEY_COMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
    }

    friend std::ostream &operator<<(std::ostream &os, const DarkNetPrediction &p)
    {
        os << p.prediction_class << " ";
        os << p.prediction_box.x << " ";
        os << p.prediction_box.y << " ";
        os << p.prediction_box.w << " ";
        os << p.prediction_box.h << " ";
        os << p.prediction_prob;
        return os;
    }
};

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

class DarkNetPredictionOutput
{
  public:
    std::vector<DarkNetPrediction> predictions;

    friend std::ostream &operator<<(std::ostream &os, const DarkNetPredictionOutput &p)
    {
        for (int i = 0; i < p.predictions.size(); i++)
        {
            os << p.predictions[i];
            if (i < p.predictions.size() - 1)
            {
                os << "\n";
            }
        }
        return os;
    }
};
}

#endif /* DARKNETPREDICTION_HPP */