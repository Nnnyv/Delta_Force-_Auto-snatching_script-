/*
 read_screen_digits_fast.cpp

 目的:
  在 Windows 平台上使用 C++ + OpenCV 快速识别屏幕指定区域内的印刷体数字（标准打印体）。
  实现尽量轻量、速度优先：不使用 Tesseract，而用基于轮廓分割 + 模板匹配的方法。
  返回 int 型（失败返回 -1）。

 思路:
  1) 使用 GDI 快速截图指定屏幕区域 -> cv::Mat (BGR)
  2) 灰度 + 二值化（速度优先，使用固定阈值或 Otsu）
  3) 轮廓检测，筛选出可能的数字区域（按面积、长宽比）
  4) 对每个候选区域与预生成的数字模板集做快速像素级比较（L1 或 L2）
  5) 将识别出的数字按 x 坐标从左到右组合成整数返回

 优点:
  - 速度较快（适用于小区域、字体规整的场景）
  - 无需外部 OCR 引擎依赖

 局限:
  - 对非常复杂背景、倾斜或严重变形的数字鲁棒性较差
  - 若数字样式与生成的模板差别较大，需调整模板生成参数或提供样本

 编译示例（假设已安装 OpenCV 并设置环境）:
  g++ -std=c++17 read_screen_digits_fast.cpp -o read_fast -I/path/to/opencv/include \
      -L/path/to/opencv/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lgdi32

 使用:
  int v = readDigitsFast(100, 200, 250, 60);
  if (v >= 0) printf("识别: %d\n", v);
  else printf("识别失败\n");

 返回:
  成功: 返回识别到的整数
  失败/未检测到数字: 返回 -1
*/
#include <windows.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <limits>

#include <opencv2/opencv.hpp>
//#include "getNum.h"

// ----------------------- 屏幕捕获 -----------------------
cv::Mat captureScreenRegion(int x, int y, int width, int height) {
    if (width <= 0 || height <= 0) return cv::Mat();

    HDC hScreenDC = GetDC(NULL);
    HDC hMemDC = CreateCompatibleDC(hScreenDC);
    HBITMAP hBitmap = CreateCompatibleBitmap(hScreenDC, width, height);
    if (!hBitmap) {
        DeleteDC(hMemDC);
        ReleaseDC(NULL, hScreenDC);
        return cv::Mat();
    }

    HGDIOBJ hOld = SelectObject(hMemDC, hBitmap);
    if (!BitBlt(hMemDC, 0, 0, width, height, hScreenDC, x, y, SRCCOPY | CAPTUREBLT)) {
        SelectObject(hMemDC, hOld);
        DeleteObject(hBitmap);
        DeleteDC(hMemDC);
        ReleaseDC(NULL, hScreenDC);
        return cv::Mat();
    }

    BITMAPINFOHEADER bi;
    ZeroMemory(&bi, sizeof(bi));
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height; // top-down
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;

    std::vector<uchar> buf(width * height * 4);
    if (!GetDIBits(hMemDC, hBitmap, 0, height, buf.data(), reinterpret_cast<BITMAPINFO*>(&bi), DIB_RGB_COLORS)) {
        SelectObject(hMemDC, hOld);
        DeleteObject(hBitmap);
        DeleteDC(hMemDC);
        ReleaseDC(NULL, hScreenDC);
        return cv::Mat();
    }

    cv::Mat matBGRA(height, width, CV_8UC4, buf.data());
    cv::Mat matBGR;
    cv::cvtColor(matBGRA, matBGR, cv::COLOR_BGRA2BGR);

    SelectObject(hMemDC, hOld);
    DeleteObject(hBitmap);
    DeleteDC(hMemDC);
    ReleaseDC(NULL, hScreenDC);

    return matBGR.clone();
}

// ----------------------- 模板生成 -----------------------
// 生成一组数字模板 (0-9) 使用 OpenCV putText（Hershey 字体），可通过调整 fontScale/thickness 更贴合目标
// 模板为二值图 (uchar 0/255)，白字(255)在黑底(0)
std::vector<cv::Mat> generateDigitTemplates(int tmplW = 28, int tmplH = 40, double fontScale = 1.0, int thickness = 2) {
    std::vector<cv::Mat> templates(10);
    int baseline = 0;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;

    for (int d = 0; d <= 9; ++d) {
        cv::Mat t = cv::Mat::zeros(tmplH, tmplW, CV_8UC1);
        std::string s = std::to_string(d);
        cv::Size textSize = cv::getTextSize(s, fontFace, fontScale, thickness, &baseline);
        // 居中绘制
        int x = (tmplW - textSize.width) / 2;
        int y = (tmplH + textSize.height) / 2;
        cv::putText(t, s, cv::Point(x, y), fontFace, fontScale, cv::Scalar(255), thickness, cv::LINE_AA);
        // 细微腐蚀/膨胀可用于调整模板粗细（按需）
        templates[d] = t;
    }
    return templates;
}

// ----------------------- 预处理与候选区域提取 -----------------------
cv::Mat preprocessFast(const cv::Mat& src) {
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    else gray = src;

    // 快速去噪
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);

    // 使用 Otsu 二值化，得到白字黑底（若屏幕为黑字白底则之后取反）
    cv::Mat bw;
    cv::threshold(blur, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 我们希望数字为白(255)，背景为黑(0)。如果当前背景为黑字白底，把它反转
    // 简单判断：图像中心像素是否为白色占多数，若是则反转
    int cx = bw.cols / 2, cy = bw.rows / 2;
    int countWhite = cv::countNonZero(bw(cv::Rect(std::max(0, cx - 5), std::max(0, cy - 5), std::min(10, bw.cols), std::min(10, bw.rows))));
    if (countWhite < 5) {
        // likely digits are dark on light background; invert so digits are white
        cv::bitwise_not(bw, bw);
    }

    // 小范围形态学操作，去噪并连接断裂的笔画
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(bw, bw, cv::MORPH_CLOSE, kernel);
    return bw;
}

struct DigitCandidate {
    cv::Rect rect;
    double score; // lower better
    int digit;    // matched digit
};

// 计算两个同尺寸二值图的归一化 L1 差别（越小越像）
double compareImagesL1(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.size() == b.size() && a.type() == b.type());
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    // diff 为 0 或 255，归一化到 [0,1]
    double s = cv::sum(diff)[0] / 255.0;
    return s / (a.rows * a.cols); // 平均每像素差异 (0..1)
}

// ----------------------- 主识别函数 -----------------------
int readDigitsFast(int x, int y, int width, int height) {
    // 截图
    cv::Mat src = captureScreenRegion(x, y, width, height);
    if (src.empty()) return -1;

    // 预处理
    cv::Mat bw = preprocessFast(src);
    if (bw.empty()) return -1;

    // 找轮廓（只取外部轮廓，加速）
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return -1;

    // 生成模板（在第一次调用时生成一次可优化；这里为简单单次生成）
    const int T_W = 28, T_H = 40;
    // 你可以调整 fontScale/thickness 以匹配屏幕上数字粗细
    std::vector<cv::Mat> templates = generateDigitTemplates(T_W, T_H, 1.0, 2);

    std::vector<DigitCandidate> candidates;

    for (const auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);

        // 过滤噪点：面积与宽高比限制
        int area = r.width * r.height;
        if (area < 20 || r.width < 6 || r.height < 10) continue;
        double aspect = (double)r.width / r.height;
        if (aspect > 1.2 && r.height < 20) {
            // 可能是小横条，不是数字或多字符连在一起，根据场景可放宽
        }

        // 裁剪并缩放到模板大小，同时保留纵横比并填充背景（中心对齐）
        cv::Mat roi = bw(r);
        cv::Mat roiResized;
        // 计算缩放尺寸
        int w = roi.cols, h = roi.rows;
        double scaleW = (double)T_W / w;
        double scaleH = (double)T_H / h;
        double scale = std::min(scaleW, scaleH);
        int newW = std::max(1, int(w * scale));
        int newH = std::max(1, int(h * scale));
        cv::resize(roi, roiResized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

        // 放到黑底模板里居中
        cv::Mat canvas = cv::Mat::zeros(T_H, T_W, CV_8UC1);
        int ox = (T_W - newW) / 2;
        int oy = (T_H - newH) / 2;
        roiResized.copyTo(canvas(cv::Rect(ox, oy, newW, newH)));

        // 比对所有模板，取最小差异
        double bestScore = 1e9;
        int bestDigit = -1;
        for (int d = 0; d <= 9; ++d) {
            double s = compareImagesL1(canvas, templates[d]);
            if (s < bestScore) {
                bestScore = s;
                bestDigit = d;
            }
        }

        // 如果最小差异太大，则认为不是数字，可调整阈值（例如 0.45）
        if (bestScore < 0.45) {
            candidates.push_back({ r, bestScore, bestDigit });
        }
    }

    if (candidates.empty()) return -1;

    // 排序按 x 坐标（从左到右）
    std::sort(candidates.begin(), candidates.end(), [](const DigitCandidate& a, const DigitCandidate& b) {
        return a.rect.x < b.rect.x;
        });

    // 将检测到的数字合并成一个整数（如果中间有间隔很大，说明可能是多组数字；这里简单合并所有）
    std::string digits;
    for (const auto& c : candidates) digits.push_back(char('0' + c.digit));

    // 去掉前导零? 根据需求决定；这里保留原样，但如果全为零，也返回 0
    // 解析为 int
    if (digits.empty()) return -1;
    // 防止过长导致 stoll 抛异常
    if (digits.size() > 10) return -1; // 超过 int 范围或者不可信

    try {
        long long val = std::stoll(digits);
        if (val < std::numeric_limits<int>::min() || val > std::numeric_limits<int>::max()) return -1;
        return static_cast<int>(val);
    }
    catch (...) {
        return -1;
    }
}



