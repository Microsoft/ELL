////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     tutorialHelpers.h
//  Authors:  Byron Changuion
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <string>
#include <vector>

#include "model.h"

class tutorialHelpers
{
public:
    struct ImagePreprocessingMetadata
    {
        std::string colorChannelOrder;
        std::vector<float> pixelScaleRange;
        std::vector<float> channelMeansRGB;
        std::vector<float> channelStdDevRGB;
    };

    static ImagePreprocessingMetadata GetImagePreprocessingMetadata(const ModelWrapper& modelWrapper)
    {
        ImagePreprocessingMetadata metadata;
        metadata.colorChannelOrder = modelWrapper.GetMetadata("model.input.expectedColorChannelOrder");

        std::string lowRangeStr = modelWrapper.GetMetadata("model.input.expectedPixelRangeLow");
        std::string highRangeStr = modelWrapper.GetMetadata("model.input.expectedPixelRangeHigh");
        if (lowRangeStr.compare("") != 0 && highRangeStr.compare("") != 0)
        {
            float lowRange = atof(lowRangeStr.c_str());
            float highRange = atof(highRangeStr.c_str());

            metadata.pixelScaleRange.push_back(lowRange);
            metadata.pixelScaleRange.push_back(highRange);
        }

        std::string redChannelMeanStr = modelWrapper.GetMetadata("model.input.redChannelMean");
        std::string greenChannelMeanStr = modelWrapper.GetMetadata("model.input.greenChannelMean");
        std::string blueChannelMeanStr = modelWrapper.GetMetadata("model.input.blueChannelMean");

        if (redChannelMeanStr.compare("") != 0 &&
            greenChannelMeanStr.compare("") != 0 &&
            blueChannelMeanStr.compare("") != 0)
        {
            float redMean = atof(redChannelMeanStr.c_str());
            float greenMean = atof(greenChannelMeanStr.c_str());
            float blueMean = atof(blueChannelMeanStr.c_str());

            metadata.channelMeansRGB.push_back(redMean);
            metadata.channelMeansRGB.push_back(greenMean);
            metadata.channelMeansRGB.push_back(blueMean);
        }

        std::string redChannelStdDevStr = modelWrapper.GetMetadata("model.input.redChannelStdDev");
        std::string greenChannelStdDevStr = modelWrapper.GetMetadata("model.input.greenChannelStdDev");
        std::string blueChannelStdDevStr = modelWrapper.GetMetadata("model.input.blueChannelStdDev");

        if (redChannelStdDevStr.compare("") != 0 &&
            greenChannelStdDevStr.compare("") != 0 &&
            blueChannelStdDevStr.compare("") != 0)
        {
            float redStdDev = atof(redChannelStdDevStr.c_str());
            float greenStdDev = atof(greenChannelStdDevStr.c_str());
            float blueStdDev = atof(blueChannelStdDevStr.c_str());

            metadata.channelStdDevRGB.push_back(redStdDev);
            metadata.channelStdDevRGB.push_back(greenStdDev);
            metadata.channelStdDevRGB.push_back(blueStdDev);
        }

        return metadata;
    }

    // Prepare an image for processing by an ELL model. Typically, this involves:
    // - Resize and center-crop to the required width and height while preserving aspect ratio.
    //   Simple resize may result in a stretched or squashed image which will affect the model's ability
    //   to classify images.
    // - Performing image preprocessing for a given dataset, including re-ordering the channels to RGB
    //   if needed (OpenCV gives the image in BGR order)
    // - Convert the OpenCV result to a std::vector<float> for use with the ELL model.
    static std::vector<float> PrepareImageForModel(cv::Mat& image, int requiredWidth, int requiredHeight, const ImagePreprocessingMetadata* preprocessingMetadata = nullptr)
    {
        cv::Mat resultImage = cv::Mat::zeros(requiredWidth, requiredHeight, image.type());

        cv::Rect roi;
        if (image.rows > image.cols) // Tall (more rows than columns)
        {
            roi.y = (image.rows - image.cols) / 2;
            roi.height = image.cols;
            roi.x = 0;
            roi.width = image.cols;
        }
        else // Wide (more columns than rows)
        {
            roi.y = 0;
            roi.height = image.rows;
            roi.x = (image.cols - image.rows) / 2;
            roi.width = image.rows;
        }

        // Center crop the image maintaining aspect ratio
        cv::Mat centerCropped = image(roi);

        // Resize to model's requirements
        cv::resize(centerCropped, resultImage, resultImage.size());

        // Convert image from 8-bit unsigned int pixel values to 32-bit float pixel values for pre-processing
        cv::Mat converted = cv::Mat::zeros(resultImage.size().width, resultImage.size().height, CV_32FC3);
        resultImage.convertTo(converted, CV_32FC3);

        // Handle preprocessing tags
        if (preprocessingMetadata != nullptr)
        {
            bool isBGRImage = true;
            if (preprocessingMetadata->colorChannelOrder.compare("rgb") == 0)
            {
                isBGRImage = false;
                cv::cvtColor(converted, converted, cv::COLOR_BGR2RGB);
            }

            if (preprocessingMetadata->pixelScaleRange.size() == 2)
            {
                // OpenCV pixel values are integers in [0, 255]
                // To scale [0, 255] to [low, high]: low + (pixel / 255.0) * (high - low)
                cv::Scalar pixelScale(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
                float lowValue = preprocessingMetadata->pixelScaleRange[0];
                float highValue = preprocessingMetadata->pixelScaleRange[1];
                converted = converted.mul(pixelScale);
                converted *= (highValue - lowValue);
                converted += lowValue;
            }

            if (preprocessingMetadata->channelMeansRGB.size() == 3)
            {
                float firstMean = 0.0f;
                float secondMean = preprocessingMetadata->channelMeansRGB[1]; // Green is second in RGB and BGR
                float thirdMean = 0.0f;
                if (isBGRImage)
                {
                    firstMean = preprocessingMetadata->channelMeansRGB[2]; // Blue first
                    thirdMean = preprocessingMetadata->channelMeansRGB[0];
                }
                else
                {
                    firstMean = preprocessingMetadata->channelMeansRGB[0]; // Red first
                    thirdMean = preprocessingMetadata->channelMeansRGB[2];
                }

                cv::Scalar means(firstMean, secondMean, thirdMean);
                converted -= means;
            }

            if (preprocessingMetadata->channelStdDevRGB.size() == 3)
            {
                float firstStdDev = 0.0f;
                float secondStdDev = preprocessingMetadata->channelStdDevRGB[1]; // Green is second in RGB and BGR
                float thirdStdDev = 0.0f;
                if (isBGRImage)
                {
                    firstStdDev = preprocessingMetadata->channelStdDevRGB[2]; // Blue first
                    thirdStdDev = preprocessingMetadata->channelStdDevRGB[0];
                }
                else
                {
                    firstStdDev = preprocessingMetadata->channelStdDevRGB[0]; // Red first
                    thirdStdDev = preprocessingMetadata->channelStdDevRGB[2];
                }

                // Divide each channel by the standard deviation == multiply each channel by the standard deviation inverse
                cv::Scalar stdDevs(1.0 / firstStdDev, 1.0 / secondStdDev, 1.0 / thirdStdDev);
                converted = converted.mul(stdDevs);
            }
        }

        // Return as a vector of floats
        // datastart and dataend give uchar* references to the data, cast to float* to get appropriate pixel values
        return { reinterpret_cast<const float*>(converted.datastart), reinterpret_cast<const float*>(converted.dataend) };
    }

    // Returns the top N scores that exceed a given threshold. The result is a vector of std::pair,
    // where the first element of each pair is the index and the second is the score.
    static std::vector<std::pair<size_t, float>> GetTopN(const std::vector<float>& predictions, size_t topN = 5, double threshold = 0.20)
    {
        // initialize original index locations
        std::vector<size_t> indexes(predictions.size());
        std::iota(indexes.begin(), indexes.end(), 0);

        // sort indexes based on comparing prediction values
        std::partial_sort(indexes.begin(), indexes.begin() + topN, indexes.end(), [&predictions](size_t index1, size_t index2) {
            return predictions[index1] > predictions[index2];
        });

        // Make a std::pair and append it to the result for N predictions
        std::vector<std::pair<size_t, float>> result;
        for (size_t i = 0; i < topN; ++i)
        {
            if (predictions[indexes[i]] >= threshold)
            {
                result.emplace_back(indexes[i], predictions[indexes[i]]);
            }
        }

        return result;
    }

    // Add a duration to a vector and calculate the mean duration.
    static double GetMeanDuration(std::vector<double>& accumulated, double duration, size_t maxAccumulatedEntries = 30)
    {
        accumulated.emplace_back(duration);

        if (accumulated.size() > maxAccumulatedEntries)
        {
            accumulated.erase(accumulated.begin());
        }

        auto meanFunction = [&accumulated](double a, double b) { return a + b / accumulated.size(); };

        return std::accumulate(accumulated.begin(), accumulated.end(), 0.0, meanFunction);
    }

    // Draw a colored text block with black text inside
    static void DrawTextBlock(cv::Mat& image, const std::string& text, cv::Point topLeft, cv::Scalar color, int height = 40)
    {
        const double fontScale = 0.7;

        cv::rectangle(image, topLeft, cv::Point(image.cols, topLeft.y + height), color, CV_FILLED);

        cv::putText(image, text, cv::Point(topLeft.x + height / 4, topLeft.y + (int)(height * 0.667)), cv::FONT_HERSHEY_COMPLEX_SMALL, fontScale, cv::Scalar(0, 0, 0), 1, CV_AA);
    }

    // Draw a header text block on an image
    static void DrawHeader(cv::Mat& image, const std::string& text)
    {
        const int blockHeight = 40;

        DrawTextBlock(image, text, cv::Point(0, 0), cv::Scalar(50, 200, 50), blockHeight);
    }

    // Draw a footer text block on an image
    static void DrawFooter(cv::Mat& image, const std::string& text)
    {
        const int blockHeight = 40;

        DrawTextBlock(image, text, cv::Point(0, image.rows - blockHeight), cv::Scalar(200, 100, 100), blockHeight);
    }
};
