#include "getNumb.h"
#include <iostream>
using namespace std;
//int main77777777() {
//    // 1) 程序启动时初始化一次（模型路径替换为你的）
//    if (!InitDigitClassifier("model.onnx", /*useCuda=*/false, /*inputSize=*/32,
//        /*binarize=*/false, /*binThr=*/0.75f)) {
//        std::cerr << "模型导入 failed.\n";
//        return 1;
//    }
//
//    // 2) 任意位置调用分类函数（例如抓取 13x19）
//    float conf = 0.0f;
//    int pred = ClassifyScreenRegion(/*x=*/100, /*y=*/100, /*w=*/13, /*h=*/19, &conf);
//    if (pred >= 0) {
//        std::cout << "Pred: " << pred << "  conf=" << conf << "\n";
//    }
//    else {
//        std::cout << "Classification failed.\n";
//    }
//    for (;;);
//    // 3) 程序退出前可选释放
//    ReleaseDigitClassifier();
//    return 0;
//}