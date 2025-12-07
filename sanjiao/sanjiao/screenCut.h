#pragma once
#include <opencv2/opencv.hpp>
#include <string>

// 只显示一次 Mat 的“静态快照”，不阻塞主线程。
// 注意：窗口刷新依赖外部周期性调用 PumpHighGUI()。
inline void ShowMat(const cv::Mat& img,
    const std::string& windowName = "Snapshot",
    bool resizable = true) {
    if (img.empty()) return;

    if (resizable) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    }
    else {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    }

    // 显示一次快照（clone 确保后续你修改 img 时窗口内容不变）
    cv::imshow(windowName, img.clone());

    // 立刻处理一次事件，确保窗口能显示出来
    cv::waitKey(1);
}

// 在主循环中定期调用，保持窗口刷新与键盘事件响应。
// delay_ms 设置为 1~10ms 即可；返回值可忽略。
inline void PumpHighGUI(int delay_ms = 1) {
    cv::waitKey(delay_ms);
}

inline void downLoad(cv::Mat& img,int num) {
	std::string n = std::to_string(num);
    bool success = cv::imwrite("D:\\sanjiao_data\\dataset\\" + n + "_number.png", img);
	if (!success) {
		std::cout << "保存失败！" << std::endl;
	}
}