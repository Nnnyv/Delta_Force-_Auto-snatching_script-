#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// 与旧程序保持一致的选项，默认输出 32×32 白底黑字
struct SP32_Options {
    bool toGray = true;
    bool denoise = false;        // 默认关闭中值滤波
    int  blurKsize = 1;          // 1 等价于不滤波
    bool useCLAHE = false;       // 默认关闭 CLAHE
    double claheClip = 1.2;
    cv::Size claheTile{ 2,2 };

    bool binarize = 1;
    bool adaptive = 0;
    int  adaptiveBlock = 7;
    int  adaptiveC = 2;

    bool morph = false;          // 默认关闭形态学
    int  openSize = 0;
    int  closeSize = 0;

    bool deskew = false;         // 屏幕截图一般不倾斜，默认关闭
    bool center = true;
    int  border = 2;

    bool outputBlackOnWhite = true; // 白底黑字
    int  targetSize = 32;
};

struct SP32_Result {
    cv::Mat viz; // uint8 单通道，保存/可视化/推理
};

// ========== 预处理核心实现（与旧逻辑一致） ==========
inline cv::Mat SP32_medianOpt(const cv::Mat& src, int k = 3, bool enable = true) {
    if (!enable || k <= 1) return src.clone();
    cv::Mat dst; cv::medianBlur(src, dst, (k | 1));
    return dst;
}

inline double SP32_estimateSkewRad(const cv::Mat& fgWhite) {
    cv::Moments mu = cv::moments(fgWhite, true); // 白=前景
    if (std::abs(mu.m00) < 1e-5) return 0.0;
    return 0.5 * std::atan2(2.0 * mu.mu11, (mu.mu20 - mu.mu02));
}

inline cv::Mat SP32_rotateKeep(const cv::Mat& src, double deg, uchar bg = 0) {
    cv::Point2f c((float)src.cols / 2.f, (float)src.rows / 2.f);
    cv::Mat M = cv::getRotationMatrix2D(c, deg, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, M, src.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(bg));
    return dst;
}

inline cv::Rect SP32_bboxNonZero(const cv::Mat& bw) {
    std::vector<cv::Point> pts; cv::findNonZero(bw, pts);
    if (pts.empty()) return { 0,0,bw.cols,bw.rows };
    return cv::boundingRect(pts);
}

inline cv::Mat SP32_centerOnSquare(const cv::Mat& img, int border, uchar bg = 0) {
    int side = (((img.cols) > (img.rows)) ? (img.cols) : (img.rows)) + (((0) > (2 * border)) ? (0) : (2 * border));
    if (side <= 0) side = 1;
    cv::Mat canvas(side, side, img.type(), cv::Scalar(bg));
    int x = (side - img.cols) / 2;
    int y = (side - img.rows) / 2;
    x = (((0) > (x)) ? (0) : (x)); y = (((0) > (y)) ? (0) : (y));
    img.copyTo(canvas(cv::Rect(x, y, img.cols, img.rows)));
    return canvas;
}

// 将输入图像按与旧程序一致的“白底黑字 32×32”流程处理
inline SP32_Result SP32_preprocessTo32x32(const cv::Mat& input, const SP32_Options& opt) {
    CV_Assert(!input.empty());
    cv::Mat gray;

    // 1) 灰度
    if (opt.toGray) {
        if (input.channels() == 1) gray = input.clone();
        else cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.channels() == 1 ? input.clone()
            : ([](const cv::Mat& m) { cv::Mat g; cv::cvtColor(m, g, cv::COLOR_BGR2GRAY); return g; })(input);
    }

    // 2) 去噪 + 可选 CLAHE
    gray = SP32_medianOpt(gray, opt.blurKsize, opt.denoise);
    if (opt.useCLAHE) {
        auto clahe = cv::createCLAHE(opt.claheClip, opt.claheTile);
        clahe->apply(gray, gray);
    }

    // 3) 二值化（尽量得到 前景=白、背景=黑）
    cv::Mat bw;
    if (opt.binarize) {
        if (opt.adaptive) {
            int blk = (((3) > (opt.adaptiveBlock | 1)) ? (3) : (opt.adaptiveBlock | 1));
            blk = (((blk) < (((((gray.cols) < (gray.rows)) ? (gray.cols) : (gray.rows)) / 2) * 2 + 1)) ? (blk) : (((((gray.cols) < (gray.rows)) ? (gray.cols) : (gray.rows)) / 2) * 2 + 1));
            if (blk < 3) blk = 3;
            cv::adaptiveThreshold(gray, bw, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                cv::THRESH_BINARY, blk, opt.adaptiveC);
        }
        else {
            cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        }
        double meanVal = cv::mean(bw)[0];
        if (meanVal > 127) cv::bitwise_not(bw, bw); // 如果是白底黑字，翻转为黑底白字
    }
    else {
        cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double meanVal = cv::mean(bw)[0];
        if (meanVal > 127) cv::bitwise_not(bw, bw);
    }
    cv::Mat fgWhite = bw; // 白=前景

    // 4) 形态学（可选）
    if (opt.morph) {
        if (opt.openSize > 0) {
            int k = 2 * opt.openSize + 1;
            cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, { k,k });
            cv::morphologyEx(fgWhite, fgWhite, cv::MORPH_OPEN, se);
        }
        if (opt.closeSize > 0) {
            int k = 2 * opt.closeSize + 1;
            cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, { k,k });
            cv::morphologyEx(fgWhite, fgWhite, cv::MORPH_CLOSE, se);
        }
    }

    // 5) 去倾斜（可选）
    if (opt.deskew) {
        double rad = SP32_estimateSkewRad(fgWhite);
        double deg = rad * 180.0 / CV_PI;
        fgWhite = SP32_rotateKeep(fgWhite, -deg, /*bg=*/0);
    }

    // 6) 裁切前景 -> 居中贴正方形（黑底）
    cv::Rect box = SP32_bboxNonZero(fgWhite);
    box &= cv::Rect(0, 0, fgWhite.cols, fgWhite.rows);
    cv::Mat cropped = fgWhite(box).clone();
    cv::Mat square = SP32_centerOnSquare(cropped, opt.border, /*bg=*/0); // 背景黑

    // 7) 极性与缩放（保存/推理用）
    cv::Mat viz = square.clone();
    if (opt.outputBlackOnWhite) {
        cv::bitwise_not(viz, viz); // 白底黑字
    }
    int interp = (opt.targetSize >= (((viz.cols) > (viz.rows)) ? (viz.cols) : (viz.rows))) ? cv::INTER_LINEAR : cv::INTER_AREA;
    cv::resize(viz, viz, cv::Size(opt.targetSize, opt.targetSize), 0, 0, interp);

    SP32_Result r; r.viz = viz;
    return r;
}

// ========== 屏幕区域截图（Win32），并套用同样预处理 ==========
#ifdef _WIN32
inline void SP32_EnableDPIAwarenessOnce() {
    static bool inited = false;
    if (inited) return;
    inited = true;

    HMODULE hUser32 = ::GetModuleHandleW(L"user32.dll");
    if (hUser32) {
        using SetDpiCtxFn = DPI_AWARENESS_CONTEXT(WINAPI*)(DPI_AWARENESS_CONTEXT);
        auto pSetDpiCtx = reinterpret_cast<SetDpiCtxFn>(GetProcAddress(hUser32, "SetThreadDpiAwarenessContext"));
        if (pSetDpiCtx) {
            pSetDpiCtx(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
            return;
        }
    }
    // 退回进程级 DPI 感知
    HMODULE hUser32b = ::GetModuleHandleW(L"user32.dll");
    if (hUser32b) {
        using SetProcDpiAwareFn = BOOL(WINAPI*)();
        auto pSetProcDpiAware = reinterpret_cast<SetProcDpiAwareFn>(GetProcAddress(hUser32b, "SetProcessDPIAware"));
        if (pSetProcDpiAware) pSetProcDpiAware();
    }
}
#endif

// 捕获屏幕矩形区域（x,y,w,h 基于物理像素），输出 BGR CV_8UC3
inline bool SP32_CaptureRegionRawBGR(int x, int y, int w, int h, cv::Mat& outBGR) {
    outBGR.release();
    if (w <= 0 || h <= 0) return false;

#ifdef _WIN32
    SP32_EnableDPIAwarenessOnce();

    HDC hScreen = ::GetDC(nullptr);
    if (!hScreen) return false;

    HDC hMemDC = ::CreateCompatibleDC(hScreen);
    if (!hMemDC) { ::ReleaseDC(nullptr, hScreen); return false; }

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = w;
    bmi.bmiHeader.biHeight = -h; // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32; // BGRA
    bmi.bmiHeader.biCompression = BI_RGB;

    void* pBits = nullptr;
    HBITMAP hDIB = ::CreateDIBSection(hScreen, &bmi, DIB_RGB_COLORS, &pBits, nullptr, 0);
    if (!hDIB || !pBits) {
        if (hDIB) ::DeleteObject(hDIB);
        ::DeleteDC(hMemDC);
        ::ReleaseDC(nullptr, hScreen);
        return false;
    }

    HGDIOBJ hOld = ::SelectObject(hMemDC, hDIB);
    BOOL ok = ::BitBlt(hMemDC, 0, 0, w, h, hScreen, x, y, SRCCOPY);
    ::SelectObject(hMemDC, hOld);

    if (!ok) {
        ::DeleteObject(hDIB);
        ::DeleteDC(hMemDC);
        ::ReleaseDC(nullptr, hScreen);
        return false;
    }

    cv::Mat bgra(h, w, CV_8UC4, pBits);
    cv::Mat bgr; cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    outBGR = bgr.clone();

    ::DeleteObject(hDIB);
    ::DeleteDC(hMemDC);
    ::ReleaseDC(nullptr, hScreen);
    return !outBGR.empty();
#else
    (void)x; (void)y; (void)w; (void)h;
    return false; // 非 Windows 平台暂不实现
#endif
}

// 一步完成：截屏区域并预处理为 32×32 白底黑字（CV_8UC1）
inline bool SP32_CaptureAndPreprocessRegion(int x, int y, int w, int h,
    const SP32_Options& opt,
    cv::Mat& out32x32) {
    out32x32.release();
    cv::Mat bgr;
    if (!SP32_CaptureRegionRawBGR(x, y, w, h, bgr)) return false;

    SP32_Result r = SP32_preprocessTo32x32(bgr, opt);
    if (r.viz.empty()) return false;

    out32x32 = r.viz.clone();
    return true;
}

// 便捷重载：cv::Rect
inline bool SP32_CaptureAndPreprocessRegion(const cv::Rect& roi,
    const SP32_Options& opt,
    cv::Mat& out32x32) {
    return SP32_CaptureAndPreprocessRegion(roi.x, roi.y, roi.width, roi.height, opt, out32x32);
}