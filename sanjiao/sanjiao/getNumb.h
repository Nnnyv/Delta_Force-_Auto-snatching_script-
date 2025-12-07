#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

struct Prediction {
    int class_index = -1;             // 0..C-1
    std::string class_name;           // 若加载了 labels.txt 则为对应标签，否则为 index 的字符串
    float confidence = 0.0f;          // top-1 概率
    std::vector<float> probabilities; // 全部类别概率，长度 C
};

class OnnxDigitClassifier {
public:
    // onnx_path: ONNX 模型路径（如 build_artifacts/model.onnx）
    // labels_path: 可选，labels.txt（每行一个类别）
    // use_cuda: 若 OpenCV 构建了 CUDA DNN，可设 true 使用 GPU
    // expected_size: 模型导出时的输入边长（你的训练默认 32）
    explicit OnnxDigitClassifier(const std::string& onnx_path,
        const std::string& labels_path = "",
        bool use_cuda = false,
        int expected_size = 32);

    // 说明：输入可以是
    // - CV_8UC1（常见）：0..255，白底黑字
    // - CV_32FC1：若为[0,1]则内部归一化到[-1,1]；若已是[-1,1]则不再归一化
    // 尺寸若非 expected_size×expected_size，会在内部 resize 到该尺寸
    Prediction predict(const cv::Mat& preprocessed);

    // 批量推理：向量内每张图类型/尺寸可不同，会统一转换
    std::vector<Prediction> predictBatch(const std::vector<cv::Mat>& batch);

    int numClasses() const { return num_classes_; }
    int expectedSize() const { return expected_size_; }

private:
    cv::dnn::Net net_;
    std::vector<std::string> labels_;
    int num_classes_ = 0;
    int expected_size_ = 32;

    static std::vector<std::string> readLines(const std::string& path);
    static void softmaxRow(const float* logits, int C, std::vector<float>& probs);
    static std::string trim(const std::string& s);

    // 将任意 CV_8UC1 或 CV_32FC1 的单通道图，统一为模型输入所需的 CV_32FC1、[-1,1]、expected_size_×expected_size_
    cv::Mat toModelInput(const cv::Mat& src) const;

    static void assertSingleChannel(const cv::Mat& m);
};