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

#include <iostream>
#include <filesystem>
#include <sstream>


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
    int side = std::max(img.cols, img.rows) + std::max(0, 2 * border);
    if (side <= 0) side = 1;
    cv::Mat canvas(side, side, img.type(), cv::Scalar(bg));
    int x = (side - img.cols) / 2;
    int y = (side - img.rows) / 2;
    x = std::max(0, x); y = std::max(0, y);
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
            int blk = std::max(3, (opt.adaptiveBlock | 1));
            int minSide = std::min(gray.cols, gray.rows);
            blk = std::min(blk, (minSide / 2) * 2 + 1);
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
    int interp = (opt.targetSize >= std::max(viz.cols, viz.rows)) ? cv::INTER_LINEAR : cv::INTER_AREA;
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

namespace fs = std::filesystem;

static void print_usage(const char* exe) {
    std::cout
        << "Usage:\n"
        << "  " << exe << " x y w h output_path [options]\n\n"
        << "Options:\n"
        << "  --denoise K            Enable median blur with kernel K (>=1, odd). Default: off\n"
        << "  --clahe                Enable CLAHE. Default: off\n"
        << "  --clahe-clip C         CLAHE clipLimit. Default: 1.2\n"
        << "  --clahe-tile WxH       CLAHE tileGridSize (e.g., 2x2). Default: 2x2\n"
        << "  --adaptive             Use adaptive threshold. Default: off\n"
        << "  --adaptive-block N     Adaptive block size (odd). Default: 7\n"
        << "  --adaptive-C C         Adaptive constant C. Default: 2\n"
        << "  --morph                Enable morphology ops. Default: off\n"
        << "  --open N               Open size radius. Default: 0\n"
        << "  --close N              Close size radius. Default: 0\n"
        << "  --deskew               Estimate skew and deskew. Default: off\n"
        << "  --center 0|1           Center to square (for compatibility). Default: 1\n"
        << "  --border N             Border pixels when centering. Default: 2\n"
        << "  --black-on-white 0|1   Output polarity. 1=white bg, black text. Default: 1\n"
        << "  --size N               Target output size (NxN). Default: 32\n"
        << std::endl;
}

static bool parse_tile(const std::string& s, int& w, int& h) {
    auto pos = s.find('x');
    if (pos == std::string::npos) return false;
    try {
        w = std::stoi(s.substr(0, pos));
        h = std::stoi(s.substr(pos + 1));
        return w > 0 && h > 0;
    }
    catch (...) { return false; }
}

int main(int argc, char** argv) {
#ifdef _WIN32
    if (argc < 6) {
        print_usage(argv[0]);
        return 1;
    }

    int x = 0, y = 0, w = 0, h = 0;
    std::string outPath;

    try {
        x = std::stoi(argv[1]);
        y = std::stoi(argv[2]);
        w = std::stoi(argv[3]);
        h = std::stoi(argv[4]);
        outPath = argv[5];
    }
    catch (...) {
        std::cerr << "Invalid numeric arguments.\n";
        print_usage(argv[0]);
        return 1;
    }

    if (w <= 0 || h <= 0) {
        std::cerr << "Width and height must be positive.\n";
        return 1;
    }

    SP32_Options opt; // defaults already match your legacy behavior

    // Parse optional args
    for (int i = 6; i < argc; ++i) {
        std::string a = argv[i];

        auto starts_with = [](const std::string& s, const std::string& p) {
            return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
            };

        if (a == "--denoise") {
            if (i + 1 >= argc) { std::cerr << "--denoise requires K\n"; return 1; }
            int k = std::stoi(argv[++i]);
            if (k < 1) k = 1;
            if ((k % 2) == 0) k += 1; // make odd
            opt.denoise = true;
            opt.blurKsize = k;
        }
        else if (a == "--clahe") {
            opt.useCLAHE = true;
        }
        else if (a == "--adaptive") {
            opt.adaptive = true;
            opt.binarize = true;
        }
        else if (a == "--morph") {
            opt.morph = true;
        }
        else if (a == "--deskew") {
            opt.deskew = true;
        }
        else if (a == "--center") {
            if (i + 1 >= argc) { std::cerr << "--center requires 0|1\n"; return 1; }
            opt.center = (std::stoi(argv[++i]) != 0);
        }
        else if (a == "--black-on-white") {
            if (i + 1 >= argc) { std::cerr << "--black-on-white requires 0|1\n"; return 1; }
            opt.outputBlackOnWhite = (std::stoi(argv[++i]) != 0);
        }
        else if (a == "--size") {
            if (i + 1 >= argc) { std::cerr << "--size requires N\n"; return 1; }
            int sz = std::stoi(argv[++i]);
            if (sz < 1) sz = 1;
            opt.targetSize = sz;
        }
        else if (a == "--clahe-clip") {
            if (i + 1 >= argc) { std::cerr << "--clahe-clip requires value\n"; return 1; }
            opt.claheClip = std::stod(argv[++i]);
        }
        else if (a == "--clahe-tile") {
            if (i + 1 >= argc) { std::cerr << "--clahe-tile requires WxH\n"; return 1; }
            int tw = 0, th = 0;
            if (!parse_tile(argv[++i], tw, th)) { std::cerr << "Invalid tile format. Use WxH\n"; return 1; }
            opt.claheTile = cv::Size(tw, th);
        }
        else if (a == "--adaptive-block") {
            if (i + 1 >= argc) { std::cerr << "--adaptive-block requires N\n"; return 1; }
            int blk = std::stoi(argv[++i]);
            if (blk < 3) blk = 3;
            if ((blk % 2) == 0) blk += 1; // odd
            opt.adaptiveBlock = blk;
        }
        else if (a == "--adaptive-C") {
            if (i + 1 >= argc) { std::cerr << "--adaptive-C requires C\n"; return 1; }
            opt.adaptiveC = std::stoi(argv[++i]);
        }
        else if (a == "--open") {
            if (i + 1 >= argc) { std::cerr << "--open requires N\n"; return 1; }
            opt.openSize = std::stoi(argv[++i]);
            if (opt.openSize < 0) opt.openSize = 0;
        }
        else if (a == "--close") {
            if (i + 1 >= argc) { std::cerr << "--close requires N\n"; return 1; }
            opt.closeSize = std::stoi(argv[++i]);
            if (opt.closeSize < 0) opt.closeSize = 0;
        }
        else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 捕获并预处理
    cv::Mat out32;
    if (!SP32_CaptureAndPreprocessRegion(x, y, w, h, opt, out32)) {
        std::cerr << "Failed to capture and preprocess the specified region.\n";
        return 2;
    }

    // 确保目录存在
    try {
        fs::path p(outPath);
        if (p.has_parent_path() && !p.parent_path().empty()) {
            fs::create_directories(p.parent_path());
        }
        // 保存（单通道 PNG）
        if (!cv::imwrite(outPath, out32)) {
            std::cerr << "Failed to save image to: " << outPath << "\n";
            return 3;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Filesystem or saving error: " << e.what() << "\n";
        return 3;
    }

    std::cout << "Saved preprocessed image to: " << outPath << "\n";
    return 0;
#else
    std::cerr << "This example only supports Windows (_WIN32) for screen capture.\n";
    return 1;
#endif
}


