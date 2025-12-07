#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <set>
#include <sstream>

struct Options {
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

    bool outputBlackOnWhite = true;
    int  targetSize = 32;
};

struct IOFlags {
    // 模式与路径
    std::filesystem::path in;        // 单文件模式输入
    std::filesystem::path out;       // 单文件模式输出
    std::filesystem::path inRoot = "D:/sanjiao_data/train";   // 目录/镜像模式输入根
    std::filesystem::path outRoot = "D:/sanjiao_data/trains"; // 目录/镜像模式输出根
    std::filesystem::path listPath;  // 列表模式

    // 其他
    bool overwrite = false;
    bool flattenList = true;         // 列表模式是否扁平保存
    std::filesystem::path configLoad; // --config
    std::filesystem::path configSave; // --save-config
    std::filesystem::path csvPath;    // --save-csv
};

struct Result {
    cv::Mat viz;    // uint8 单通道，保存/可视化
};

static cv::Mat medianOpt(const cv::Mat& src, int k = 3, bool enable = true) {
    if (!enable || k <= 1) return src.clone();
    cv::Mat dst; cv::medianBlur(src, dst, (k | 1));
    return dst;
}

static double estimateSkewRad(const cv::Mat& fgWhite) {
    cv::Moments mu = cv::moments(fgWhite, true); // 白=前景
    if (std::abs(mu.m00) < 1e-5) return 0.0;
    return 0.5 * std::atan2(2.0 * mu.mu11, (mu.mu20 - mu.mu02));
}

static cv::Mat rotateKeep(const cv::Mat& src, double deg, uchar bg = 0) {
    cv::Point2f c((float)src.cols / 2.f, (float)src.rows / 2.f);
    cv::Mat M = cv::getRotationMatrix2D(c, deg, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, M, src.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(bg));
    return dst;
}

static cv::Rect bboxNonZero(const cv::Mat& bw) {
    std::vector<cv::Point> pts; cv::findNonZero(bw, pts);
    if (pts.empty()) return { 0,0,bw.cols,bw.rows };
    return cv::boundingRect(pts);
}

static cv::Mat centerOnSquare(const cv::Mat& img, int border, uchar bg = 0) {
    int side = std::max(img.cols, img.rows) + std::max(0, 2 * border);
    if (side <= 0) side = 1;
    cv::Mat canvas(side, side, img.type(), cv::Scalar(bg));
    int x = (side - img.cols) / 2;
    int y = (side - img.rows) / 2;
    x = std::max(0, x); y = std::max(0, y);
    img.copyTo(canvas(cv::Rect(x, y, img.cols, img.rows)));
    return canvas;
}

static Result preprocessDigit13x19(const cv::Mat& input, const Options& opt) {
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

    // 2) 去噪（可关/弱化） + 可选 CLAHE
    gray = medianOpt(gray, opt.blurKsize, opt.denoise);
    if (opt.useCLAHE) {
        auto clahe = cv::createCLAHE(opt.claheClip, opt.claheTile);
        clahe->apply(gray, gray);
    }

    // 3) 二值化：尽量得到“前景=白、背景=黑”的图
    cv::Mat bw;
    if (opt.binarize) {
        if (opt.adaptive) {
            int blk = std::max(3, opt.adaptiveBlock | 1);
            blk = std::min(blk, (std::min(gray.cols, gray.rows) / 2) * 2 + 1); // 防止过大
            if (blk < 3) blk = 3;
            cv::adaptiveThreshold(gray, bw, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                cv::THRESH_BINARY, blk, opt.adaptiveC);
        }
        else {
            cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        }
        // 若白占比太高（白底黑字），翻转为黑底白字再继续
        double meanVal = cv::mean(bw)[0];
        if (meanVal > 127) cv::bitwise_not(bw, bw);
    }
    else {
        cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double meanVal = cv::mean(bw)[0];
        if (meanVal > 127) cv::bitwise_not(bw, bw);
    }
    cv::Mat fgWhite = bw; // 白=前景

    // 4) 形态学（可关/弱化）
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

    // 5) 去倾斜（可关）
    if (opt.deskew) {
        double rad = estimateSkewRad(fgWhite);
        double deg = rad * 180.0 / CV_PI;
        fgWhite = rotateKeep(fgWhite, -deg, /*bg=*/0);
    }

    // 6) 裁切前景 -> 居中贴正方形（黑底）
    cv::Rect box = bboxNonZero(fgWhite);
    box &= cv::Rect(0, 0, fgWhite.cols, fgWhite.rows);
    cv::Mat cropped = fgWhite(box).clone();
    cv::Mat square = centerOnSquare(cropped, opt.border, /*bg=*/0); // 背景黑

    // 7) 极性与缩放（保存用）
    cv::Mat viz = square.clone();
    if (opt.outputBlackOnWhite) {
        cv::bitwise_not(viz, viz); // 白底黑字
    }
    int interp = (opt.targetSize >= std::max(viz.cols, viz.rows)) ? cv::INTER_LINEAR : cv::INTER_AREA;
    cv::resize(viz, viz, cv::Size(opt.targetSize, opt.targetSize), 0, 0, interp);

    Result r; r.viz = viz;
    return r;
}

// ---------- 配置文件（YAML）读写 ----------
static void saveConfigYAML(const Options& opt, const IOFlags& io, const std::filesystem::path& path) {
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    fs << "options" << "{"
        << "toGray" << opt.toGray
        << "denoise" << opt.denoise
        << "blurKsize" << opt.blurKsize
        << "useCLAHE" << opt.useCLAHE
        << "claheClip" << opt.claheClip
        << "claheTileW" << opt.claheTile.width
        << "claheTileH" << opt.claheTile.height
        << "binarize" << opt.binarize
        << "adaptive" << opt.adaptive
        << "adaptiveBlock" << opt.adaptiveBlock
        << "adaptiveC" << opt.adaptiveC
        << "morph" << opt.morph
        << "openSize" << opt.openSize
        << "closeSize" << opt.closeSize
        << "deskew" << opt.deskew
        << "center" << opt.center
        << "border" << opt.border
        << "outputBlackOnWhite" << opt.outputBlackOnWhite
        << "targetSize" << opt.targetSize
        << "}";
    fs << "io" << "{"
        << "in" << io.in.string()
        << "out" << io.out.string()
        << "inRoot" << io.inRoot.string()
        << "outRoot" << io.outRoot.string()
        << "listPath" << io.listPath.string()
        << "overwrite" << (int)io.overwrite
        << "flattenList" << (int)io.flattenList
        << "csvPath" << io.csvPath.string()
        << "}";
}

static void loadConfigYAML(Options& opt, IOFlags& io, const std::filesystem::path& path) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) return;
    cv::FileNode o = fs["options"];
    if (!o.empty()) {
        o["toGray"] >> opt.toGray;
        o["denoise"] >> opt.denoise;
        o["blurKsize"] >> opt.blurKsize;
        o["useCLAHE"] >> opt.useCLAHE;
        o["claheClip"] >> opt.claheClip;
        int w = opt.claheTile.width, h = opt.claheTile.height;
        o["claheTileW"] >> w; o["claheTileH"] >> h; opt.claheTile = cv::Size(w, h);
        o["binarize"] >> opt.binarize;
        o["adaptive"] >> opt.adaptive;
        o["adaptiveBlock"] >> opt.adaptiveBlock;
        o["adaptiveC"] >> opt.adaptiveC;
        o["morph"] >> opt.morph;
        o["openSize"] >> opt.openSize;
        o["closeSize"] >> opt.closeSize;
        o["deskew"] >> opt.deskew;
        o["center"] >> opt.center;
        o["border"] >> opt.border;
        o["outputBlackOnWhite"] >> opt.outputBlackOnWhite;
        o["targetSize"] >> opt.targetSize;
    }
    cv::FileNode i = fs["io"];
    if (!i.empty()) {
        std::string s;
        i["in"] >> s; if (!s.empty()) io.in = s;
        i["out"] >> s; if (!s.empty()) io.out = s;
        i["inRoot"] >> s; if (!s.empty()) io.inRoot = s;
        i["outRoot"] >> s; if (!s.empty()) io.outRoot = s;
        i["listPath"] >> s; if (!s.empty()) io.listPath = s;
        int v = 0;
        i["overwrite"] >> v; io.overwrite = (v != 0);
        i["flattenList"] >> v; io.flattenList = (v != 0);
        i["csvPath"] >> s; if (!s.empty()) io.csvPath = s;
    }
}

// ---------- 参数解析辅助 ----------
static bool parseWxH(const std::string& s, cv::Size& out) {
    auto pos = s.find('x');
    if (pos == std::string::npos) return false;
    int w = std::stoi(s.substr(0, pos));
    int h = std::stoi(s.substr(pos + 1));
    if (w <= 0 || h <= 0) return false;
    out = cv::Size(w, h);
    return true;
}

// ---------- CSV ----------
static void ensureCSVWithHeader(const std::filesystem::path& csvPath) {
    if (csvPath.empty()) return;
    if (std::filesystem::exists(csvPath)) return;
    std::ofstream ofs(csvPath);
    ofs << "input,output,status,width,height\n";
}

static void appendCSV(const std::filesystem::path& csvPath,
    const std::string& in, const std::string& out,
    const std::string& status, int w, int h) {
    if (csvPath.empty()) return;
    std::ofstream ofs(csvPath, std::ios::app);
    ofs << '"' << in << '"' << ','
        << '"' << out << '"' << ','
        << status << ','
        << w << ',' << h << '\n';
}

// ---------- 处理并保存（单张） ----------
static bool processAndSaveOne(const std::filesystem::path& inPath,
    const std::filesystem::path& outPath,
    const Options& opt, bool overwrite,
    const std::filesystem::path& csvPath) {
    cv::Mat src = cv::imread(inPath.string(), cv::IMREAD_UNCHANGED);
    if (src.empty()) {
        std::cerr << "Failed to read: " << inPath << "\n";
        appendCSV(csvPath, inPath.string(), outPath.string(), "read_fail", 0, 0);
        return false;
    }
    if (!overwrite && std::filesystem::exists(outPath)) {
        appendCSV(csvPath, inPath.string(), outPath.string(), "skipped_exists", 0, 0);
        return true;
    }
    std::error_code ec;
    std::filesystem::create_directories(outPath.parent_path(), ec);

    Result r = preprocessDigit13x19(src, opt);
    if (!cv::imwrite(outPath.string(), r.viz)) {
        std::cerr << "Failed to write: " << outPath << "\n";
        appendCSV(csvPath, inPath.string(), outPath.string(), "write_fail", 0, 0);
        return false;
    }
    appendCSV(csvPath, inPath.string(), outPath.string(), "ok", r.viz.cols, r.viz.rows);
    return true;
}

int mainllllllll(int argc, char** argv) {
    // 降低 OpenCV 日志噪声
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    Options opt;
    IOFlags io;

    // 先扫描 --config 以加载默认
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--config=", 0) == 0) io.configLoad = a.substr(9);
    }
    if (!io.configLoad.empty()) {
        loadConfigYAML(opt, io, io.configLoad);
    }

    // 正式解析参数（命令行覆盖配置）
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout <<
                "Usage:\n"
                "  Single file:  --in=img.png --out=out.png [options]\n"
                "  Mirror dir:   --in-root=dataset/train --out-root=dataset/trains [options]\n"
                "  From list:    --list=images.txt --out-root=out [--in-root=base] [--flatten-list=0|1]\n"
                "Common options:\n"
                "  --size=32  --no-deskew  --white-on-black  --no-binarize  --adaptive\n"
                "  --no-denoise | --blur=1  --no-clahe | --clahe-clip=1.2 | --clahe-tile=2x2\n"
                "  --no-morph | --open=0 --close=0  --overwrite\n"
                "Config & logs:\n"
                "  --config=opt.yml  --save-config=opt.yml  --save-csv=report.csv\n";
            return 0;
        }
        // 模式与路径
        else if (a.rfind("--in=", 0) == 0)         io.in = a.substr(5);
        else if (a.rfind("--out=", 0) == 0)        io.out = a.substr(6);
        else if (a.rfind("--in-root=", 0) == 0)    io.inRoot = a.substr(10);
        else if (a.rfind("--out-root=", 0) == 0)   io.outRoot = a.substr(11);
        else if (a.rfind("--list=", 0) == 0)       io.listPath = a.substr(7);
        else if (a.rfind("--flatten-list=", 0) == 0) { int v = std::stoi(a.substr(15)); io.flattenList = (v != 0); }

        // 处理强度
        else if (a.rfind("--size=", 0) == 0)       opt.targetSize = std::stoi(a.substr(7));
        else if (a == "--no-deskew")              opt.deskew = false;

        else if (a == "--no-denoise")             opt.denoise = false;
        else if (a.rfind("--blur=", 0) == 0) { opt.blurKsize = std::max(1, std::stoi(a.substr(7))); if ((opt.blurKsize & 1) == 0) ++opt.blurKsize; }
        else if (a == "--no-clahe")               opt.useCLAHE = false;
        else if (a.rfind("--clahe-clip=", 0) == 0) opt.claheClip = std::max(0.0, std::stod(a.substr(13)));
        else if (a.rfind("--clahe-tile=", 0) == 0) { cv::Size s; if (parseWxH(a.substr(13), s)) opt.claheTile = s; }

        else if (a == "--no-morph")               opt.morph = false;
        else if (a.rfind("--open=", 0) == 0)       opt.openSize = std::max(0, std::stoi(a.substr(7)));
        else if (a.rfind("--close=", 0) == 0)      opt.closeSize = std::max(0, std::stoi(a.substr(8)));

        else if (a == "--white-on-black")         opt.outputBlackOnWhite = false;
        else if (a == "--no-binarize")            opt.binarize = false;
        else if (a == "--adaptive")               opt.adaptive = true;

        // 覆盖、配置、日志
        else if (a == "--overwrite")              io.overwrite = true;
        else if (a.rfind("--save-config=", 0) == 0)io.configSave = a.substr(14);
        else if (a.rfind("--save-csv=", 0) == 0)   io.csvPath = a.substr(11);
        else if (a.rfind("--config=", 0) == 0) { /* 已处理 */ }
    }

    if (!io.configSave.empty()) {
        saveConfigYAML(opt, io, io.configSave);
        std::cout << "Saved config: " << io.configSave << "\n";
    }
    if (!io.csvPath.empty()) ensureCSVWithHeader(io.csvPath);

    // 判断模式：优先单文件；否则列表；否则目录镜像
    const std::set<std::string> exts = { ".png",".jpg",".jpeg",".bmp",".tif",".tiff" };

    // 单文件模式
    if (!io.in.empty() && !io.out.empty()) {
        bool ok = processAndSaveOne(io.in, io.out, opt, io.overwrite, io.csvPath);
        return ok ? 0 : 2;
    }

    // 列表模式
    if (!io.listPath.empty()) {
        std::ifstream ifs(io.listPath);
        if (!ifs) {
            std::cerr << "Failed to open list: " << io.listPath << "\n";
            return 1;
        }
        size_t nOK = 0, nFail = 0, idx = 0;
        std::string line;
        while (std::getline(ifs, line)) {
            std::string s = line;
            // 去掉首尾空白
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isspace(c); }));
            s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) { return !std::isspace(c); }).base(), s.end());
            if (s.empty() || s[0] == '#') continue;

            std::filesystem::path inPath = s;
            if (!std::filesystem::exists(inPath)) { std::cerr << "Not found: " << inPath << "\n"; ++nFail; continue; }

            std::filesystem::path dst;
            if (!io.inRoot.empty() && std::filesystem::exists(io.inRoot) && !io.flattenList) {
                std::error_code ec;
                std::filesystem::path rel = std::filesystem::relative(inPath, io.inRoot, ec);
                if (ec) rel = inPath.filename();
                dst = io.outRoot / rel;
            }
            else {
                // 扁平保存：使用文件名，若重名，加序号后缀
                dst = io.outRoot / inPath.filename();
                if (!io.overwrite && std::filesystem::exists(dst)) {
                    // 加索引避免覆盖
                    std::filesystem::path stem = dst.stem();
                    std::filesystem::path ext = dst.extension();
                    do {
                        ++idx;
                        dst = io.outRoot / (stem.string() + "_" + std::to_string(idx) + ext.string());
                    } while (std::filesystem::exists(dst));
                }
            }
            bool ok = processAndSaveOne(inPath, dst, opt, io.overwrite, io.csvPath);
            ok ? ++nOK : ++nFail;
        }
        std::cout << "Done (list). ok=" << nOK << " fail=" << nFail
            << "\nOutput root: " << io.outRoot << "\n";
        return (nFail == 0) ? 0 : 2;
    }

    // 目录镜像模式（默认）
    if (!std::filesystem::exists(io.inRoot)) {
        std::cerr << "Source root not found: " << io.inRoot << "\n";
        return 1;
    }
    std::error_code ec;
    std::filesystem::create_directories(io.outRoot, ec);

    size_t nFiles = 0, nFail = 0;
    for (auto it = std::filesystem::recursive_directory_iterator(io.inRoot);
        it != std::filesystem::recursive_directory_iterator(); ++it) {
        if (!it->is_regular_file()) continue;
        auto ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (!exts.count(ext)) continue;

        std::filesystem::path rel = std::filesystem::relative(it->path(), io.inRoot, ec);
        if (ec) rel = it->path().filename();
        std::filesystem::path dst = io.outRoot / rel;

        bool ok = processAndSaveOne(it->path(), dst, opt, io.overwrite, io.csvPath);
        ok ? ++nFiles : ++nFail;
    }
    std::cout << "Done (mirror). saved=" << nFiles << " fail=" << nFail
        << "\nOutput root: " << std::filesystem::weakly_canonical(io.outRoot, ec).string() << "\n";
    return (nFail == 0) ? 0 : 2;
}