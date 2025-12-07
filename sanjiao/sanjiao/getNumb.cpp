#include "getNumb.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

std::string OnnxDigitClassifier::trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::vector<std::string> OnnxDigitClassifier::readLines(const std::string& path) {
    std::vector<std::string> lines;
    if (path.empty()) return lines;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return lines;
    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (!line.empty()) lines.emplace_back(line);
    }
    return lines;
}

void OnnxDigitClassifier::softmaxRow(const float* logits, int C, std::vector<float>& probs) {
    probs.resize(C);
    float maxv = logits[0];
    for (int i = 1; i < C; ++i) maxv = std::max(maxv, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < C; ++i) {
        probs[i] = std::exp(static_cast<double>(logits[i] - maxv));
        sum += probs[i];
    }
    if (sum <= 0.0) sum = 1.0;
    for (int i = 0; i < C; ++i) probs[i] = static_cast<float>(probs[i] / sum);
}

void OnnxDigitClassifier::assertSingleChannel(const cv::Mat& m) {
    if (m.empty()) throw std::invalid_argument("Input image is empty.");
    if (m.channels() != 1) throw std::invalid_argument("Input must be single-channel (grayscale).");
}

cv::Mat OnnxDigitClassifier::toModelInput(const cv::Mat& src) const {
    assertSingleChannel(src);

    // 1) 转为 float32
    cv::Mat f32;
    if (src.type() == CV_8UC1) {
        src.convertTo(f32, CV_32F, 1.0 / 255.0); // [0,1]
    }
    else if (src.type() == CV_32FC1) {
        f32 = src.clone();
    }
    else {
        throw std::invalid_argument("Unsupported input type. Use CV_8UC1 or CV_32FC1.");
    }

    // 2) 自适应归一化到 [-1,1]（与训练 Normalize(mean=0.5,std=0.5) 等价）
    double minv = 0.0, maxv = 1.0;
    cv::minMaxLoc(f32, &minv, &maxv);
    if (minv >= 0.0 && maxv <= 1.0) {
        // [0,1] -> [-1,1]
        f32 = f32 * 2.0f - 1.0f;
    }
    else if (minv >= -1.0 && maxv <= 1.0) {
        // 已是 [-1,1]，不处理
    }
    else if (minv >= 0.0 && maxv <= 255.0) {
        // 可能是 [0,255] 的 float
        f32 = f32 * (1.0f / 255.0f);
        f32 = f32 * 2.0f - 1.0f;
    }
    else {
        // 非常规范围，按 [0,1] 处理
        f32 = f32 * 2.0f - 1.0f;
    }

    // 3) 尺寸规范化到 expected_size_×expected_size_
    if (f32.rows != expected_size_ || f32.cols != expected_size_) {
        int interp = (expected_size_ >= std::max(f32.cols, f32.rows)) ? cv::INTER_LINEAR : cv::INTER_AREA;
        cv::resize(f32, f32, cv::Size(expected_size_, expected_size_), 0, 0, interp);
    }

    return f32; // CV_32FC1, [-1,1], expected_size_^2
}

OnnxDigitClassifier::OnnxDigitClassifier(const std::string& onnx_path,
    const std::string& labels_path,
    bool use_cuda,
    int expected_size)
    : expected_size_(expected_size) {
    net_ = cv::dnn::readNet(onnx_path);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load ONNX model: " + onnx_path);
    }

    // 后端/设备选择
    try {
#if (CV_VERSION_MAJOR >= 4)
        if (use_cuda) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        else {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
#else
        (void)use_cuda;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif
    }
    catch (...) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    labels_ = readLines(labels_path);

    // 尝试从网络输出推断类别数
    if (!labels_.empty()) {
        num_classes_ = static_cast<int>(labels_.size());
    }
    else {
        cv::Mat dummy(expected_size_, expected_size_, CV_32FC1, cv::Scalar(0));
        cv::Mat blob = cv::dnn::blobFromImage(dummy, 1.0, cv::Size(), cv::Scalar(), false, false);
        net_.setInput(blob);
        std::vector<std::string> outs = net_.getUnconnectedOutLayersNames();
        cv::Mat out = outs.empty() ? net_.forward() : net_.forward(outs[0]);

        if (out.dims == 2) {
            num_classes_ = out.size[1];
        }
        else if (out.dims == 4 && out.size[2] == 1 && out.size[3] == 1) {
            num_classes_ = out.size[1];
        }
        else if (out.dims == 1) {
            num_classes_ = out.size[0];
        }
        else {
            num_classes_ = 0;
        }
    }
}

Prediction OnnxDigitClassifier::predict(const cv::Mat& preprocessed) {
    cv::Mat f32 = toModelInput(preprocessed);
    cv::Mat blob = cv::dnn::blobFromImage(f32, 1.0, cv::Size(), cv::Scalar(), false, false);
    net_.setInput(blob);

    std::vector<std::string> outs = net_.getUnconnectedOutLayersNames();
    cv::Mat out = outs.empty() ? net_.forward() : net_.forward(outs[0]);

    // 统一为 [N, C]
    cv::Mat scores2d;
    int N = 1, C = 0;
    if (out.dims == 2) {
        N = out.size[0]; C = out.size[1]; scores2d = out;
    }
    else if (out.dims == 4 && out.size[2] == 1 && out.size[3] == 1) {
        N = out.size[0]; C = out.size[1]; scores2d = out.reshape(1, N);
    }
    else if (out.dims == 1) {
        N = 1; C = out.size[0]; scores2d = out.reshape(1, 1);
    }
    else {
        N = 1; C = static_cast<int>(out.total()); scores2d = out.reshape(1, 1);
    }

    if (num_classes_ == 0) num_classes_ = C;

    const float* logits = scores2d.ptr<float>(0);
    Prediction pred;
    softmaxRow(logits, C, pred.probabilities);
    auto it = std::max_element(pred.probabilities.begin(), pred.probabilities.end());
    pred.class_index = static_cast<int>(std::distance(pred.probabilities.begin(), it));
    pred.confidence = *it;
    if (!labels_.empty() && pred.class_index >= 0 && pred.class_index < (int)labels_.size()) {
        pred.class_name = labels_[pred.class_index];
    }
    else {
        pred.class_name = std::to_string(pred.class_index);
    }
    return pred;
}

std::vector<Prediction> OnnxDigitClassifier::predictBatch(const std::vector<cv::Mat>& batch) {
    if (batch.empty()) return {};

    std::vector<cv::Mat> converted;
    converted.reserve(batch.size());
    for (const auto& m : batch) {
        converted.emplace_back(toModelInput(m));
    }

    cv::Mat blob = cv::dnn::blobFromImages(converted, 1.0, cv::Size(), cv::Scalar(), false, false);
    net_.setInput(blob);

    std::vector<std::string> outs = net_.getUnconnectedOutLayersNames();
    cv::Mat out = outs.empty() ? net_.forward() : net_.forward(outs[0]);

    // 统一为 [N, C]
    cv::Mat scores2d;
    int N = 1, C = 0;
    if (out.dims == 2) {
        N = out.size[0]; C = out.size[1]; scores2d = out;
    }
    else if (out.dims == 4 && out.size[2] == 1 && out.size[3] == 1) {
        N = out.size[0]; C = out.size[1]; scores2d = out.reshape(1, N);
    }
    else if (out.dims == 1) {
        N = 1; C = out.size[0]; scores2d = out.reshape(1, 1);
    }
    else {
        N = out.size[0]; C = static_cast<int>(out.total() / N); scores2d = out.reshape(1, N);
    }

    if (num_classes_ == 0) num_classes_ = C;

    std::vector<Prediction> results;
    results.reserve(N);
    for (int i = 0; i < N; ++i) {
        const float* logits = scores2d.ptr<float>(i);
        Prediction pred;
        softmaxRow(logits, C, pred.probabilities);
        auto it = std::max_element(pred.probabilities.begin(), pred.probabilities.end());
        pred.class_index = static_cast<int>(std::distance(pred.probabilities.begin(), it));
        pred.confidence = *it;
        if (!labels_.empty() && pred.class_index >= 0 && pred.class_index < (int)labels_.size()) {
            pred.class_name = labels_[pred.class_index];
        }
        else {
            pred.class_name = std::to_string(pred.class_index);
        }
        results.emplace_back(std::move(pred));
    }
    return results;
}