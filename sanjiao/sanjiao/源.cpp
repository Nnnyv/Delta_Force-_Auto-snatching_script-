#include"getNumb.h"
#include"getSpace.h"
#include"click.h"
#include"preProcess.h"
#include"screenCut.h"
#include <iostream>
#include<vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <atomic>
#include <thread>
#include <conio.h>  // Windows键盘输入
#include <windows.h>
#include <queue>

using namespace std;
OnnxDigitClassifier model("model.onnx", "", false, 32);
//1000 300  交易行收藏界面，第一个物品
//2310 1100 交易行最大购买数量
//2200 1220 点击购买
////////////////////225 1245 145 30 查看最低价格

//242 1245 17 27 百万第一位
//257 1245 17 27
//271 1245 16 27

//2195 1155 16 25   右方百万第一位
//2210 1155 16 25   右方百万第二位
//2224 1155 16 25

std::atomic<bool> programRunning(true);

// 键盘监听线程函数
//void escTerminationListener() {
//	while (programRunning) {
//		if (_kbhit()) {  // 检测键盘输入
//			int ch = _getch();  // 获取按键
//			if (ch == 27) {     // ESC键的ASCII码是27
//				programRunning = false;
//				std::cout << "\nESC键按下，程序即将终止..." << std::endl;
//				break;
//			}
//		}
//		std::this_thread::sleep_for(std::chrono::milliseconds(100));
//	}
//}

// 只需在主程序中插入这一条语句
//#define ENABLE_ESC_TERMINATION std::thread([](){ escTerminationListener(); }).detach();
deque<int> nums;
int TIMES = 30;
int AVERAGE=-1;
int BROUGHT = 0;
void getNum(int x, int y, int w, int h, int& n, float& conf) {
	SP32_Options opt;
	cv::Mat out;
	SP32_CaptureAndPreprocessRegion(x, y, w, h, opt, out);
	Prediction predict;
	predict=model.predict(out);
	n = predict.class_index;
	conf = predict.confidence;
	//ShowMat(out, "123123");
	//if (iff_load!=-1)downLoad(out, iff_load);
	return;
}

int getNumAll() {
	int n;
	float conf;
	getNum(2195, 1155, 16, 25, n, conf);
	if (conf < 0.95f)return -1;
	int num1 = n;
	getNum(2210, 1155 ,16, 25, n, conf);
	if (conf < 0.95f)return -1;
	int num2 = n;
	getNum(2224, 1155, 16, 25, n, conf);
	if (conf < 0.95f)return -1;
	int num3 = n;
	return num1 * 100 + num2 * 10 + num3;
}

int numsAverage() {
	AVERAGE = 0;
	deque<int> Nums = nums;
	sort(Nums.begin(), Nums.end());
	int out = TIMES / 4;
	for (int i = out; i < TIMES - out; ++i) {
		AVERAGE += nums[i];
	}
	AVERAGE /= TIMES - 2 * out;
	return AVERAGE;
}
void getAverage() {
	//int TIMES = 30;
	//vector<int> nums;
	nums.resize(TIMES);
	int n;
	AVERAGE = 0;
	for (int i = 0; i < TIMES; ++i) {
		HumanLikeMouseClick(1000+rand()%3, 300+rand()%3,13+rand()%3, MB_LEFT, 1, 13 + rand() % 2, false);
		Sleep(320 + rand() % 3);
		n = getNumAll();
		cout << "Get Num: " << n << endl;
		if (n == -1)--i;
		else nums[i] = n;
		SendEscKey(8+rand()%3);
		Sleep(150 + rand() % 3);
	}
	deque<int> Nums = nums;
	sort(Nums.begin(), Nums.end());
	int out = TIMES / 4;
	for (int i = out; i < TIMES - out; ++i) {
		AVERAGE += nums[i];
	}
	AVERAGE /= TIMES-2*out;

}

int main_getScreen() {
	Sleep(2000);
	if (SetProcessDpiAwarenessContext) {
		// Win10+: 优先用 Per-monitor v2
		SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
	}
	else {
		// 兼容老系统的回退
		SetProcessDPIAware();
	}
	system("cls");
	int x= 271,
		y=1245,
		w=16,
		h=27;
	const float thr = 0.95f;
	ScreenRectOverlay* pi;
	int N = -1;
	float Conf = -1;
	int download = 0;
	int begin = 0;
	for (;;) {
		pi = new ScreenRectOverlay(x, y, w, h);
		pi->free();
		getNum(x, y, w, h, N, Conf);
		cout << "N: " << N << " Conf: " << Conf << endl;
		pi = new ScreenRectOverlay(x, y, w, h);
		cin >> begin;
		if (begin==123123)break;
		pi->free();
		Sleep(500);
	}
	pi->free();
	for (download= 0;download<10000;++download) {
		getNum(x, y, w, h, N, Conf);
		cout << "N: " << N << " Conf: " << Conf <<" Num:" <<download<< endl;
		//Sleep(5);
	}

	return 0;
}

void aRound(const int& cha) {
	int n;
	HumanLikeMouseClick(1000, 300,13,MB_LEFT,1,10,false);
	Sleep(320);
	n = getNumAll();
	if (n == -1) {
		cout << "Read Error!" << endl;
		SendEscKey(10 + rand() % 3);
		Sleep(50 + rand() % 3);
		return;
	}
	cout << "Now Num: " << n << " Average: " << AVERAGE << " Cha: " << AVERAGE-n << endl;
	if (AVERAGE - n >= cha) {
		//HumanLikeMouseClick(2310, 1100, 13, MB_LEFT, 1, 11, false);
		//Sleep(60 + rand() % 5);
		HumanLikeMouseClick(2200, 1220, 12, MB_LEFT, 1, 12, false);
		Sleep(60 + rand() % 5);
		cout << "Buyed!" << endl;
		exit(0);
		BROUGHT += 9999;
	}
	else {
		cout << "Not Buyed!" << endl;
	}
	//HumanLikeMouseClick(2310, 1100, 3, MB_LEFT, 1, 11, false);
	SendEscKey(11 + rand() % 3);
	Sleep(180 + rand() % 3);
	nums.pop_front();
	nums.push_back(n);
	numsAverage();
	cout << "New Average Price: " << AVERAGE  << endl;
	return;
	
}

int main() {
	if (SetProcessDpiAwarenessContext) {
		// Win10+: 优先用 Per-monitor v2
		SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
	}
	else {
		// 兼容老系统的回退
		SetProcessDPIAware();
	}
	EnsureDpiAwareOnce();
	system("cls");

	srand((unsigned int)time(NULL));
	cout << "输入差值:";
	int cha;
	cin >> cha;
	cout << endl;
	//ENABLE_ESC_TERMINATION;
	Sleep(3000);
	cout << "Start!" << endl;
	//ScreenRectOverlay* pi;
	getAverage();
	cout << "Average Price: " << AVERAGE << endl;
	Sleep(500);
	/////return 0;//////

	for (int i=0;i<100;++i) {
		aRound(cha);
		if (BROUGHT >= 20)break;
	}

	return 0;
}


int maiklklkln() {
	ScreenRectOverlay* pi;
	int x = 2200,
		y = 1220,
		w = 28,
		h = 20;
	int n;
	float conf;
	for (;;) {
		getNum(x, y, w, h,n,conf);
		cout << "N: " << n << " Conf: " << conf << endl;
		pi = new ScreenRectOverlay(x, y, w, h);
		cin >> x >> y >> w >> h;
		pi->free();
	}
}