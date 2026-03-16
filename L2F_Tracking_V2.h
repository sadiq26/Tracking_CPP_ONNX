// L2F_Tracking_V2.h : Include file for standard system include files,
// or project specific include files.

#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include<onnxruntime_cxx_api.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <utility>

#include "utils.h"

// TODO: Reference additional headers your program requires here.
using namespace std;




class YOLODetector
{
public:
    explicit YOLODetector(std::nullptr_t) {};
    YOLODetector(const std::string& modelPath,
        const bool& isGPU,
        const cv::Size& inputSize);

    std::vector<Detection> detect(cv::Mat& image, const float& confThreshold, const float& iouThreshold);

private:
    Ort::Env env{ nullptr };
    Ort::SessionOptions sessionOptions{ nullptr };
    Ort::Session session{ nullptr };

    void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
        const cv::Size& originalImageShape,
        std::vector<Ort::Value>& outputTensors,
        const float& confThreshold, const float& iouThreshold);

    static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
        float& bestConf, int& bestClassId);

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    vector<std::string> inputNameStrings;
    vector<std::string> outputNameStrings;
    bool isDynamicInputShape{};
    cv::Size2f inputImageShape;

};