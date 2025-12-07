#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// -------------------- Template generation --------------------
struct Template {
    char ch; // '0'..'9' or ','
    Mat img; // binary image (white-on-black)
};

static vector<Template> generateTemplates(int tmplW = 40, int tmplH = 64)
{
    vector<Template> templates;
    string chars = "0123456789,";
    vector<double> scales = { 0.9, 1.0, 1.1 };
    vector<int> thicknesses = { 1, 2, 3 };

    for (char c : chars) {
        for (double scale : scales) {
            for (int th : thicknesses) {
                Mat canvas = Mat::zeros(tmplH, tmplW, CV_8UC1);
                int fontFace = FONT_HERSHEY_SIMPLEX;
                string s(1, c);

                int baseline = 0;
                Size textSize = getTextSize(s, fontFace, scale, th, &baseline);
                Point org((tmplW - textSize.width) / 2, (tmplH + textSize.height) / 2);

                putText(canvas, s, org, fontFace, scale, Scalar(255), th, LINE_AA);

                Mat bin;
                threshold(canvas, bin, 127, 255, THRESH_BINARY);

                Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
                morphologyEx(bin, bin, MORPH_CLOSE, kernel);

                Template t;
                t.ch = c;
                t.img = bin;
                templates.push_back(t);
            }
        }
    }
    return templates;
}

// Resize keeping aspect ratio to fit into (w,h) and pad with black
static Mat resizeAndPad(const Mat& src, int w, int h)
{
    Mat dst;
    int src_w = src.cols, src_h = src.rows;
    double scale = min((double)w / src_w, (double)h / src_h);
    int nw = max(1, (int)round(src_w * scale));
    int nh = max(1, (int)round(src_h * scale));
    resize(src, dst, Size(nw, nh));
    Mat out = Mat::zeros(h, w, src.type());
    int x = (w - nw) / 2;
    int y = (h - nh) / 2;
    dst.copyTo(out(Rect(x, y, nw, nh)));
    return out;
}

// -------------------- Matching function --------------------
static char matchChar(const Mat& roiBin, const vector<Template>& templates, double& outScore)
{
    if (templates.empty()) {
        outScore = 0.0;
        return '?';
    }

    int tmplW = templates[0].img.cols;
    int tmplH = templates[0].img.rows;
    Mat candidate = resizeAndPad(roiBin, tmplW, tmplH);

    Mat candBin;
    if (candidate.channels() > 1) cvtColor(candidate, candBin, COLOR_BGR2GRAY);
    else candBin = candidate;
    threshold(candBin, candBin, 127, 255, THRESH_BINARY);

    char best = '?';
    double bestScore = -1.0;
    Mat f1;
    candBin.convertTo(f1, CV_32F, 1.0 / 255.0);

    for (const Template& t : templates) {
        Mat f2;
        t.img.convertTo(f2, CV_32F, 1.0 / 255.0);

        Scalar mean1, std1, mean2, std2;
        meanStdDev(f1, mean1, std1);
        meanStdDev(f2, mean2, std2);
        Mat n1 = f1 - mean1[0];
        Mat n2 = f2 - mean2[0];
        double denom = std1[0] * std2[0] * (double)(n1.total());
        double num = (double)(n1.dot(n2));
        double score = 0.0;
        if (denom > 1e-6) score = num / denom;
        else score = (num > 0) ? 1.0 : 0.0;

        if (score > bestScore) {
            bestScore = score;
            best = t.ch;
        }
    }

    outScore = bestScore;
    return best;
}

// -------------------- Image recognition pipeline --------------------
string getImageNumber(const Mat& inputImg)
{
    if (inputImg.empty()) return "";

    Mat img = inputImg.clone();
    Mat gray;

    // Ensure single-channel CV_8U gray image
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }
    else if (img.channels() == 4) {
        // PNG with alpha often -> convert to gray properly
        cvtColor(img, gray, COLOR_BGRA2GRAY);
    }
    else if (img.channels() == 1) {
        gray = img;
    }
    else {
        // fallback: convert to 8U then to gray
        Mat tmp;
        img.convertTo(tmp, CV_8U);
        if (tmp.channels() == 4) cvtColor(tmp, gray, COLOR_BGRA2GRAY);
        else if (tmp.channels() == 3) cvtColor(tmp, gray, COLOR_BGR2GRAY);
        else gray = tmp;
    }

    if (gray.empty()) return "";

    // Force type to CV_8UC1 if it's not already
    if (gray.type() != CV_8UC1) {
        gray.convertTo(gray, CV_8U);
    }

    // upscale if small (improves recognition)
    int minH = 80;
    if (gray.rows < minH) {
        double upscale = (double)minH / gray.rows;
        resize(gray, gray, Size(), upscale, upscale, INTER_CUBIC);
    }

    // Try adaptive threshold (white text on dark bg -> THRESH_BINARY_INV after)
    Mat thr;
    adaptiveThreshold(gray, thr, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 15, 8);

    // Small morphology open to remove noise
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    morphologyEx(thr, thr, MORPH_OPEN, kernel);

    // Find contours
    vector<vector<Point>> contours;
    findContours(thr, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    struct BBox { Rect r; double area; };
    vector<BBox> boxes;
    for (auto& c : contours) {
        Rect r = boundingRect(c);
        double area = contourArea(c);
        if (r.height < 0.30 * thr.rows) continue;
        if (r.width < 4) continue;
        if (r.width > 0.95 * thr.cols) continue;
        boxes.push_back({ r, area });
    }

    if (boxes.empty()) {
        // fallback to Otsu
        Mat otsu;
        threshold(gray, otsu, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
        findContours(otsu, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        boxes.clear();
        for (auto& c : contours) {
            Rect r = boundingRect(c);
            double area = contourArea(c);
            if (r.height < 0.3 * otsu.rows) continue;
            if (r.width < 4) continue;
            boxes.push_back({ r, area });
        }
        if (boxes.empty()) return "";
        thr = otsu;
    }

    // Merge boxes that are close horizontally
    sort(boxes.begin(), boxes.end(), [](const BBox& a, const BBox& b) { return a.r.x < b.r.x; });
    vector<Rect> merged;
    for (size_t i = 0; i < boxes.size(); ++i) {
        Rect cur = boxes[i].r;
        if (!merged.empty()) {
            Rect& last = merged.back();
            int gap = cur.x - (last.x + last.width);
            if (gap < 0 || gap < (int)(0.15 * thr.rows)) {
                last = last | cur;
                continue;
            }
        }
        merged.push_back(cur);
    }

    static vector<Template> templates = generateTemplates(40, 64);

    sort(merged.begin(), merged.end(), [](const Rect& a, const Rect& b) { return a.x < b.x; });
    string result;
    for (const Rect& r : merged) {
        if (r.width <= 0 || r.height <= 0) continue;
        int pad = max(2, (int)(0.05 * thr.rows));
        int nx = max(0, r.x - pad);
        int ny = max(0, r.y - pad);
        int nw = min(thr.cols - nx, r.width + 2 * pad);
        int nh = min(thr.rows - ny, r.height + 2 * pad);
        if (nw <= 0 || nh <= 0) continue;
        Mat roiPad = thr(Rect(nx, ny, nw, nh));

        double score = 0.0;
        char ch = matchChar(roiPad, templates, score);
        if (score < 0.25) {
            if ((double)r.width / r.height < 0.5) {
                result.push_back(',');
            }
            else {
                result.push_back('?');
            }
        }
        else {
            result.push_back(ch);
        }
    }

    return result;
}

int ertertertmain(int argc, char** argv)
{
    

    string path;
    cin >> path;
    Mat img = imread(path, IMREAD_UNCHANGED);
    if (img.empty()) {
        cerr << "Failed to load image: " << path << "\n";
        return 2;
    }

    Mat proc = img;
    if (argc >= 6) {
        int x = atoi(argv[2]);
        int y = atoi(argv[3]);
        int w = atoi(argv[4]);
        int h = atoi(argv[5]);
        Rect roi(x, y, w, h);
        Rect imgRect(0, 0, img.cols, img.rows);
        roi = roi & imgRect;
        if (roi.width <= 0 || roi.height <= 0) {
            cerr << "Invalid ROI\n";
            return 3;
        }
        proc = img(roi).clone();
    }

    string out = getImageNumber(proc);
    cout << out << endl;
    return 0;
}