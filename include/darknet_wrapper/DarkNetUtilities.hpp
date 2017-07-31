/* 
 * Copyright (C) 2017 daniele de gregorio, University of Bologna - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GNU GPLv3 license.
 *
 * please write to: d.degregorio@unibo.it
 */

#ifndef DARKNETUTILITIES_HPP
#define DARKNETUTILITIES_HPP

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

namespace darknet_wrapper
{
class DarkNetUtilities
{
  public:
    static image ipl_to_image(IplImage *src)
    {
        unsigned char *data = (unsigned char *)src->imageData;
        int h = src->height;
        int w = src->width;
        int c = src->nChannels;
        int step = src->widthStep;
        image out = make_image(w, h, c);
        if (!out.data)
        {
            printf("@ ipl_to_image, out.data is NULL\n");
            exit(-1);
        }
        int i, j, k, count = 0;
        ;

        for (k = 0; k < c; ++k)
        {
            for (i = 0; i < h; ++i)
            {
                for (j = 0; j < w; ++j)
                {
                    out.data[count++] = data[i * step + j * c + k] / 255.;
                }
            }
        }
        return out;
    }

    static void image_to_ipl(image p, IplImage *&ipl)
    {
        int x, y, k;
        image copy = copy_image(p);
        constrain_image(copy);
        if (p.c == 3)
            rgbgr_image(copy);

        IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
        int step = disp->widthStep;
        for (y = 0; y < p.h; ++y)
        {
            for (x = 0; x < p.w; ++x)
            {
                for (k = 0; k < p.c; ++k)
                {
                    disp->imageData[y * step + x * p.c + k] = (unsigned char)(get_pixel(copy, x, y, k) * 255);
                }
            }
        }
        free_image(copy);
        ipl = disp;
    }

    static std::map<std::string, cv::Scalar> colors_map;
    static std::vector<std::string> colors_names;
    static void initColors()
    {
        colors_map["pink"] = cv::Scalar(233, 30, 99);
        colors_map["blue"] = cv::Scalar(33, 150, 243);
        colors_map["indigo"] = cv::Scalar(63, 81, 181);
        colors_map["brown"] = cv::Scalar(121, 85, 72);
        colors_map["purple"] = cv::Scalar(156, 39, 176);
        colors_map["grey"] = cv::Scalar(158, 158, 158);
        colors_map["yellow"] = cv::Scalar(255, 235, 59);
        colors_map["teal"] = cv::Scalar(0, 150, 136);
        colors_map["cyan"] = cv::Scalar(0, 188, 212);
        colors_map["orange"] = cv::Scalar(255, 152, 0);
        colors_map["green"] = cv::Scalar(76, 175, 80);
        colors_map["lime"] = cv::Scalar(205, 220, 57);
        colors_map["amber"] = cv::Scalar(255, 193, 7);
        colors_map["red"] = cv::Scalar(244, 67, 54);

        for (std::map<std::string, cv::Scalar>::iterator it = colors_map.begin(); it != colors_map.end(); ++it)
        {
            colors_names.push_back(it->first);
        }
    }

    static cv::Scalar getColor(int index)
    {
        if (colors_map.size() == 0)
        {
            DarkNetUtilities::initColors();
        }
        index = index % int(colors_names.size());
        return colors_map[colors_names[index]];
    }
};

std::map<std::string, cv::Scalar> DarkNetUtilities::colors_map;
std::vector<std::string> DarkNetUtilities::colors_names;
}

#endif /* DARKNETUTILITIES_HPP */