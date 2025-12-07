#pragma once
#include <windows.h>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>
#include <iostream>
class ScreenRectOverlay {
public:
    ScreenRectOverlay(int x, int y, int w, int h)
        : x_(x), y_(y), w_(w), h_(h), hwnd_(NULL), running_(true)
    {
        // 启动窗口线程并等待 hwnd 返回或失败
        std::promise<HWND> p;
        ready_ = p.get_future();
        th_ = std::thread(&ScreenRectOverlay::threadProc, this, std::move(p));
        // 等待窗口创建（超时保护）
        if (ready_.valid()) {
            auto status = ready_.wait_for(std::chrono::milliseconds(500));
            if (status == std::future_status::ready) {
                hwnd_ = ready_.get();
            }
            else {
                // 超时：仍尝试从 future 获取（如果线程稍后创建会阻塞到创建）
                try { hwnd_ = ready_.get(); }
                catch (...) { hwnd_ = NULL; }
            }
        }
    }

    // 不可拷贝
    ScreenRectOverlay(const ScreenRectOverlay&) = delete;
    ScreenRectOverlay& operator=(const ScreenRectOverlay&) = delete;

    // 可移动（简单实现）
    ScreenRectOverlay(ScreenRectOverlay&& other) noexcept
        : x_(other.x_), y_(other.y_), w_(other.w_), h_(other.h_),
        hwnd_(other.hwnd_), running_(other.running_.load())
    {
        th_ = std::move(other.th_);
        other.hwnd_ = NULL;
        other.running_ = false;
    }

    ScreenRectOverlay& operator=(ScreenRectOverlay&&) = delete;

    ~ScreenRectOverlay() {
        Close();
    }
    void free() {
        delete(this);
    }
    // 显式关闭
    void Close() {
        if (!running_) return;
        running_ = false;
        if (hwnd_) {
            // 请求关闭窗口（会导致消息循环退出）
            PostMessage(hwnd_, WM_CLOSE, 0, 0);
        }
        if (th_.joinable()) th_.join();
        hwnd_ = NULL;
    }

private:
    int x_, y_, w_, h_;
    HWND hwnd_;
    std::thread th_;
    std::future<HWND> ready_;
    std::atomic<bool> running_;

    // 窗口过程（静态）
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        switch (msg) {
        case WM_CREATE: {
            // lpCreateParams 是传入的 this 指针
            CREATESTRUCT* pcs = reinterpret_cast<CREATESTRUCT*>(lp);
            SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pcs->lpCreateParams));
            return 0;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            // 用透明色键背景（会被 SetLayeredWindowAttributes 设为透明）
            // 背景填充为洋红色 (255,0,255)
            HBRUSH bg = CreateSolidBrush(RGB(255, 0, 255));
            RECT rc;
            GetClientRect(hwnd, &rc);
            FillRect(hdc, &rc, bg);
            DeleteObject(bg);

            // 从 user data 拿到实例，读取边框参数
            ScreenRectOverlay* self = reinterpret_cast<ScreenRectOverlay*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
            if (self) {
                // 画绿色边框
                int thickness = (((2) > ((((6) < ((((1) > ((self->w_ + self->h_) / 200)) ? (1) : ((self->w_ + self->h_) / 200)))) ? (6) : ((((1) > ((self->w_ + self->h_) / 200)) ? (1) : ((self->w_ + self->h_) / 200)))))) ? (2) : ((((6) < ((((1) > ((self->w_ + self->h_) / 200)) ? (1) : ((self->w_ + self->h_) / 200)))) ? (6) : ((((1) > ((self->w_ + self->h_) / 200)) ? (1) : ((self->w_ + self->h_) / 200)))))); // 简单自适应
                HPEN pen = CreatePen(PS_SOLID, thickness, RGB(0, 255, 0));
                HGDIOBJ oldPen = SelectObject(hdc, pen);
                HBRUSH oldBrush = (HBRUSH)SelectObject(hdc, GetStockObject(NULL_BRUSH));
                Rectangle(hdc, rc.left + thickness / 2, rc.top + thickness / 2, rc.right - thickness / 2, rc.bottom - thickness / 2);
                SelectObject(hdc, oldBrush);
                SelectObject(hdc, oldPen);
                DeleteObject(pen);
            }

            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;
        default:
            return DefWindowProc(hwnd, msg, wp, lp);
        }
    }

    // 线程过程：注册类、创建窗口并运行消息循环
    void threadProc(std::promise<HWND> prom) {
        const wchar_t* clsName = L"ScreenRectOverlay_Class";

        WNDCLASSEXW wcl = { 0 };
        wcl.cbSize = sizeof(wcl);
        wcl.style = CS_HREDRAW | CS_VREDRAW;
        wcl.lpfnWndProc = ScreenRectOverlay::WndProc;
        wcl.hInstance = GetModuleHandle(NULL);
        wcl.hbrBackground = NULL;
        wcl.lpszClassName = clsName;

        RegisterClassExW(&wcl);

        // 创建无边框弹出窗口（位置与大小就是目标区域）
        // 使用 WS_POPUP; 扩展样式包括分层窗口、穿透、置顶、不出现在任务栏
        DWORD exStyle = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW;
        HWND hwnd = CreateWindowExW(
            exStyle,
            clsName,
            L"ScreenRectOverlayWindow",
            WS_POPUP,
            x_, y_, w_, h_,
            NULL,
            NULL,
            GetModuleHandle(NULL),
            this // lpCreateParams -> 用于 WM_CREATE 存储指针
        );

        if (!hwnd) {
            prom.set_value(NULL);
            return;
        }

        // 使得指定颜色（magenta）透明
        // 注意：必须在创建后调用 SetLayeredWindowAttributes 才生效
        SetLayeredWindowAttributes(hwnd, RGB(255, 0, 255), 0, LWA_COLORKEY);

        // 显示窗口但不激活
        ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        UpdateWindow(hwnd);

        // 返回 hwnd 给创建线程
        prom.set_value(hwnd);

        // 消息循环
        MSG msg;
        while (running_) {
            BOOL ret = GetMessage(&msg, NULL, 0, 0);
            if (ret == -1) break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // 注销窗口类（可选）
        UnregisterClassW(clsName, GetModuleHandle(NULL));
    }
};

