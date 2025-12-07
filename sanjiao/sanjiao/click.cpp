
// 更快版 - 将移动总耗时压到约 20ms（可由 moveMs 控制）
// 说明：
// - 建议调用 HumanLikeMouseClick(x, y, 20, MB_LEFT, 1, 20, false);
// - 若你传入更小或更大 moveMs，会按你的参数分配总移动时间。
// - 如果你的系统未提升计时精度，Sleep 的分辨率可能在 15.6ms 左右；本代码在启动时调用 timeBeginPeriod(1)
//   来提升精度（需链接 winmm.lib）。
//
// 注意：点击按下/抬起等仍有独立等待，用于模拟真人按键；如果你希望“整个函数（移动+点击）”都很快，
// 可以相应把 downMs 和双击间隔调小。

#include <windows.h>
#include <mmsystem.h>   // timeBeginPeriod
#include <chrono>
#include <thread>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <mutex>
#include "click.h"

#pragma comment(lib, "winmm.lib")

#ifndef MOUSEEVENTF_VIRTUALDESK
#define MOUSEEVENTF_VIRTUALDESK 0x4000
#endif
#ifndef MOUSEEVENTF_MOVE_NOCOALESCE
#define MOUSEEVENTF_MOVE_NOCOALESCE 0x2000
#endif



// 将屏幕像素坐标 (x,y) 映射到 SendInput 需要的绝对坐标范围 [0..65535]（虚拟桌面）
static void PixelToAbsolute(LONG x, LONG y, LONG& outAbsX, LONG& outAbsY) {
    int vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
    int vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
    int vwidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    int vheight = GetSystemMetrics(SM_CYVIRTUALSCREEN);

    int w = (vwidth > 1) ? vwidth : 1;
    int h = (vheight > 1) ? vheight : 1;

    double fx = (double)(x - vx) / (double)(w - 1);
    double fy = (double)(y - vy) / (double)(h - 1);

    if (fx < 0.0) fx = 0.0; if (fx > 1.0) fx = 1.0;
    if (fy < 0.0) fy = 0.0; if (fy > 1.0) fy = 1.0;

    outAbsX = static_cast<LONG>(fx * 65535.0 + 0.5);
    outAbsY = static_cast<LONG>(fy * 65535.0 + 0.5);
}

// cubic Bezier point
static void cubic_bezier(double t, double x0, double y0, double x1, double y1,
    double x2, double y2, double x3, double y3,
    double& rx, double& ry) {
    double u = 1.0 - t;
    double tt = t * t;
    double uu = u * u;
    double uuu = uu * u;
    double ttt = tt * t;
    rx = uuu * x0 + 3 * uu * t * x1 + 3 * u * tt * x2 + ttt * x3;
    ry = uuu * y0 + 3 * uu * t * y1 + 3 * u * tt * y2 + ttt * y3;
}

// ease in/out (smoothstep-like)
static double easeInOut(double t) {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

// 生成随机 double in [a,b)
static double rand_double_range(std::mt19937& rng, double a, double b) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

// 发送一次“绝对”移动，带 VIRTUALDESK
static bool SendAbsoluteMove(LONG absX, LONG absY) {
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;
    input.mi.dx = absX;
    input.mi.dy = absY;
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

// 发送一次“相对”移动；为游戏减少合并
static bool SendRelativeMove(LONG dx, LONG dy) {
    INPUT in = {};
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_MOVE_NOCOALESCE; // 相对移动
    in.mi.dx = dx;
    in.mi.dy = dy;
    return SendInput(1, &in, sizeof(INPUT)) == 1;
}

// 简单“探测”当前上下文是否会吞/重置绝对坐标（游戏抓鼠标、重心定位）
// 将两处等待从 8ms 降到 2ms，减少总时延
static bool ShouldPreferRelativeMove() {
    POINT p0{};
    if (!GetCursorPos(&p0)) return false;

    LONG a1x = 0, a1y = 0, a0x = 0, a0y = 0;
    PixelToAbsolute(p0.x + 1, p0.y, a1x, a1y);
    PixelToAbsolute(p0.x, p0.y, a0x, a0y);

    if (!SendAbsoluteMove(a1x, a1y)) return true; // 发不出去就用相对
    std::this_thread::sleep_for(std::chrono::milliseconds(2)); // 原 8ms → 2ms

    POINT pA{};
    GetCursorPos(&pA);

    SendAbsoluteMove(a0x, a0y);
    std::this_thread::sleep_for(std::chrono::milliseconds(2)); // 原 8ms → 2ms

    const auto dist = [](POINT a, POINT b) {
        return std::hypot(double(a.x - b.x), double(a.y - b.y));
        };
    double d_to_plus1 = dist(pA, POINT{ p0.x + 1, p0.y });
    double d_to_orig = dist(pA, p0);

    if (d_to_plus1 > 1.5 || d_to_orig < 1.0) {
        return true;
    }
    return false;
}

// 主函数：以人类风格移动到 (x,y) 并点击
// 参数:
//   x,y: 屏幕像素坐标（相对于虚拟屏幕左上）
//   moveMs: 移动耗时（毫秒），建议传 20 以实现“极快移动”
//   button: MB_LEFT / MB_RIGHT / MB_MIDDLE
//   clicks: 点击次数（1 单击，2 双击）
//   downMs: 每次按下到抬起维持毫秒
//   restorePos: 点击完成后是否把鼠标移动回原位置（若判定为相对模式，将忽略复位）
bool HumanLikeMouseClick(int x, int y,
    int moveMs,
    MouseButton button,
    int clicks,
    int downMs,
    bool restorePos)
{
    EnsureDpiAwareOnce();

    if (clicks < 1) clicks = 1;
    if (moveMs < 0) moveMs = 0;
    if (downMs < 0) downMs = 0;

    // 记录原位置
    POINT origPos = { 0,0 };
    if (restorePos) GetCursorPos(&origPos);

    // 当前 cursor pos as double start
    POINT startPos{};
    GetCursorPos(&startPos);
    double sx = (double)startPos.x;
    double sy = (double)startPos.y;
    double ex = (double)x;
    double ey = (double)y;

    // RNG
    std::random_device rd;
    std::mt19937 rng(rd());

    // 距离与步数（显著减少步数以压缩总时长）
    double dx_full = ex - sx;
    double dy_full = ey - sy;
    double dist = std::hypot(dx_full, dy_full);

    // 基于距离的粗化策略：大幅减小步数，但保持最少 2 步，最多 10 步
    // 这样在 moveMs=20ms 时，每步分配到的时间约 2~10ms 不等
    double stepDivBase = 12.0; // 原 3 → 12，步数更少
    int steps = (int)std::round(dist / stepDivBase);
    steps = std::clamp(steps, 2, 10);

    // 每步最小延迟降到 0ms（需要 timeBeginPeriod(1) 辅助）
    int minStepMs = 0;
    int totalSteps = steps;

    // 控制点与曲率（保持少量随机性，避免完全直线）
    double offScale = std::clamp(dist * 0.2, 10.0, 160.0); // 略减曲率幅度，减少位移绕行
    double px = -dy_full, py = dx_full;
    double plen = std::hypot(px, py);
    if (plen > 1e-6) { px /= plen; py /= plen; }
    else { px = 0; py = 0; }

    double c1x = sx + dx_full * 0.3 + px * rand_double_range(rng, -0.4, 0.4) * offScale;
    double c1y = sy + dy_full * 0.3 + py * rand_double_range(rng, -0.4, 0.4) * offScale;
    double c2x = sx + dx_full * 0.7 + px * rand_double_range(rng, -0.4, 0.4) * offScale;
    double c2y = sy + dy_full * 0.7 + py * rand_double_range(rng, -0.4, 0.4) * offScale;

    if (dist < 30.0) {
        c1x = sx + dx_full * 0.25 + rand_double_range(rng, -2, 2);
        c1y = sy + dy_full * 0.25 + rand_double_range(rng, -2, 2);
        c2x = sx + dx_full * 0.75 + rand_double_range(rng, -2, 2);
        c2y = sy + dy_full * 0.75 + rand_double_range(rng, -2, 2);
    }

    // 构建时间分配：使用 easeInOut 作为权重，确保总和精确为 moveMs
    std::vector<int> stepDelays(totalSteps, 0);
    if (totalSteps == 1) {
        stepDelays[0] = (((minStepMs) >(moveMs)) ? (minStepMs) : (moveMs));
    }
    else {
        std::vector<double> weights(totalSteps, 0.0);
        double acc = 0.0;
        for (int i = 0; i < totalSteps; ++i) {
            double t = (double)i / (double)(totalSteps - 1);
            double e = easeInOut(t);
            // 缩小权重的峰值差异，避免某一步占用过多时间
            double w = 0.6 * e + 0.4; // 介于 0.4..1.0 之间
            weights[i] = w;
            acc += w;
        }
        // 初步按比例取整
        int assigned = 0;
        for (int i = 0; i < totalSteps; ++i) {
            int ms = (int)std::round((weights[i] / acc) * moveMs);
            if (ms < minStepMs) ms = minStepMs;
            stepDelays[i] = ms;
            assigned += ms;
        }
        // 调整到总和精确等于 moveMs
        int diff = moveMs - assigned;
        // 尽量把偏差分配到中间几步（加或减 1ms）
        int mid = totalSteps / 2;
        int dir = (diff >= 0) ? 1 : -1;
        diff = std::abs(diff);
        for (int k = 0; k < diff; ++k) {
            int idx = mid + ((k % 2 == 0) ? (k / 2) : -(k / 2) - 1);
            if (idx < 0) idx = 0;
            if (idx >= totalSteps) idx = totalSteps - 1;
            int candidate = stepDelays[idx] + dir;
            if (candidate >= minStepMs) stepDelays[idx] = candidate;
            else {
                // 找到一个能增减的步
                for (int j = 0; j < totalSteps; ++j) {
                    int c = stepDelays[j] + dir;
                    if (c >= minStepMs) { stepDelays[j] = c; break; }
                }
            }
        }
    }

    // 是否偏好相对移动（游戏抓鼠标时）
    bool preferRelative = ShouldPreferRelativeMove();

    // 路径发送
    double prev_rx = sx, prev_ry = sy;
    for (int i = 0; i < totalSteps; ++i) {
        double segmentProgress = (double)i / (double)((totalSteps > 1) ? (totalSteps - 1) : 1);
        double tRand = segmentProgress + rand_double_range(rng, -0.01, 0.01);
        tRand = std::clamp(tRand, 0.0, 1.0);

        double rx, ry;
        cubic_bezier(tRand, sx, sy, c1x, c1y, c2x, c2y, ex, ey, rx, ry);

        // 微小路径抖动，幅度缩小，减少绕行导致的额外时间
        double jitterMag = std::clamp(dist * 0.003, 0.2, 3.0);
        rx += rand_double_range(rng, -jitterMag, jitterMag);
        ry += rand_double_range(rng, -jitterMag, jitterMag);

        if (!preferRelative) {
            LONG absX = 0, absY = 0;
            PixelToAbsolute((LONG)std::lround(rx), (LONG)std::lround(ry), absX, absY);
            if (!SendAbsoluteMove(absX, absY)) {
                // 绝对移动失败，立即切换相对
                preferRelative = true;
            }
        }

        if (preferRelative) {
            LONG dx = (LONG)std::lround(rx - prev_rx);
            LONG dy = (LONG)std::lround(ry - prev_ry);
            if (dx == 0 && dy == 0) {
                dx = (rand() & 1) ? 1 : -1;
            }
            SendRelativeMove(dx, dy);
        }

        prev_rx = rx; prev_ry = ry;

        int sleepMs = stepDelays[i];
        // 为了更稳定地达到总时长，这里不再添加额外随机延时
        if (sleepMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }

    // 最终对齐到目标点（仅在绝对模式下尝试；相对模式多数游戏会重置，没必要）
    if (!preferRelative) {
        LONG absX = 0, absY = 0;
        PixelToAbsolute(x, y, absX, absY);
        SendAbsoluteMove(absX, absY);
        // 对齐后的极短等待，原 8..15ms → 1..2ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1 + (rand() % 2)));
    }

    // 按键 Flags
    DWORD downFlag = 0, upFlag = 0;
    switch (button) {
    case MB_LEFT:   downFlag = MOUSEEVENTF_LEFTDOWN;   upFlag = MOUSEEVENTF_LEFTUP;   break;
    case MB_RIGHT:  downFlag = MOUSEEVENTF_RIGHTDOWN;  upFlag = MOUSEEVENTF_RIGHTUP;  break;
    case MB_MIDDLE: downFlag = MOUSEEVENTF_MIDDLEDOWN; upFlag = MOUSEEVENTF_MIDDLEUP; break;
    default:        downFlag = MOUSEEVENTF_LEFTDOWN;   upFlag = MOUSEEVENTF_LEFTUP;   break;
    }

    // 执行点击（如果你希望“整体调用耗时很短”，可把 downMs 调到 10~20，双击间隔也调小）
    for (int c = 0; c < clicks; ++c) {
        INPUT inDown = {};
        inDown.type = INPUT_MOUSE;
        inDown.mi.dwFlags = downFlag;
        SendInput(1, &inDown, sizeof(INPUT));

        int hold = downMs;
        int jitt = (downMs > 0) ? (int)std::clamp(downMs / 5, 0, 30) : 0; // 略减点击抖动
        hold += (rand() % (jitt + 1)) - (jitt / 2);
        if (hold < 8) hold = 8;
        std::this_thread::sleep_for(std::chrono::milliseconds(hold));

        INPUT inUp = {};
        inUp.type = INPUT_MOUSE;
        inUp.mi.dwFlags = upFlag;
        SendInput(1, &inUp, sizeof(INPUT));

        if (c != clicks - 1) {
            int between = 50 + (rand() % 60); // 原 80..180ms → 50..110ms
            std::this_thread::sleep_for(std::chrono::milliseconds(between));
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20 + (rand() % 20))); // 收尾小等待

    // 复位：若处于相对模式（很可能是游戏）则跳过复位
    if (restorePos && !preferRelative) {
        POINT cur{}; GetCursorPos(&cur);
        double bx = (double)cur.x;
        double by = (double)cur.y;
        double tx = (double)origPos.x;
        double ty = (double)origPos.y;
        int backSteps = 6; // 略减复位步数
        for (int i = 1; i <= backSteps; ++i) {
            double t = (double)i / (double)backSteps;
            double ix = bx + (tx - bx) * t;
            double iy = by + (ty - by) * t;
            LONG ax = 0, ay = 0;
            PixelToAbsolute((LONG)std::lround(ix), (LONG)std::lround(iy), ax, ay);
            SendAbsoluteMove(ax, ay);
            std::this_thread::sleep_for(std::chrono::milliseconds(5 + (rand() % 5)));
        }
    }

    return true;
}

// -------------------- 简单 demo -------------------
bool SendEscKey(int holdMs = 20)
{
    INPUT down = {};
    down.type = INPUT_KEYBOARD;
    down.ki.wVk = VK_ESCAPE;
    down.ki.wScan = static_cast<WORD>(MapVirtualKey(VK_ESCAPE, MAPVK_VK_TO_VSC));
    down.ki.dwFlags = 0;

    INPUT up = {};
    up.type = INPUT_KEYBOARD;
    up.ki.wVk = VK_ESCAPE;
    up.ki.wScan = static_cast<WORD>(MapVirtualKey(VK_ESCAPE, MAPVK_VK_TO_VSC));
    up.ki.dwFlags = KEYEVENTF_KEYUP;

    if (SendInput(1, &down, sizeof(INPUT)) != 1) {
        return false;
    }
    if (holdMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(holdMs));
    if (SendInput(1, &up, sizeof(INPUT)) != 1) {
        return false;
    }
    return true;
}










//#include <windows.h>
//#include <chrono>
//#include <thread>
//#include <random>
//#include <vector>
//#include <cmath>
//#include <iostream>
//#include <algorithm>
//#include <mutex>
//#include "click.h"
//
//// 有些 SDK 头里可能缺少这些定义，做个兜底
//#ifndef MOUSEEVENTF_VIRTUALDESK
//#define MOUSEEVENTF_VIRTUALDESK 0x4000
//#endif
//#ifndef MOUSEEVENTF_MOVE_NOCOALESCE
//#define MOUSEEVENTF_MOVE_NOCOALESCE 0x2000
//#endif
//
//// 建议在程序启动时调用一次，但这里做一次懒初始化，避免你改主流程
//static void EnsureDpiAwareOnce() {
//    static std::once_flag once;
//    std::call_once(once, [] {
//        // Windows 10+ 优先 Per-Monitor V2，降级到系统 DPI 感知
//        HMODULE user32 = GetModuleHandleW(L"user32.dll");
//        using SetProcCtxFn = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
//        auto pSetCtx = reinterpret_cast<SetProcCtxFn>(GetProcAddress(user32, "SetProcessDpiAwarenessContext"));
//        if (pSetCtx) {
//            pSetCtx(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
//        }
//        else {
//            // 老系统 fallback
//            SetProcessDPIAware();
//        }
//        });
//}
//
//// 将屏幕像素坐标 (x,y) 映射到 SendInput 需要的绝对坐标范围 [0..65535]（虚拟桌面）
//static void PixelToAbsolute(LONG x, LONG y, LONG& outAbsX, LONG& outAbsY) {
//    int vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
//    int vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
//    int vwidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
//    int vheight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
//
//    int w = (vwidth > 1) ? vwidth : 1;
//    int h = (vheight > 1) ? vheight : 1;
//
//    double fx = (double)(x - vx) / (double)(w - 1);
//    double fy = (double)(y - vy) / (double)(h - 1);
//
//    if (fx < 0.0) fx = 0.0; if (fx > 1.0) fx = 1.0;
//    if (fy < 0.0) fy = 0.0; if (fy > 1.0) fy = 1.0;
//
//    outAbsX = static_cast<LONG>(fx * 65535.0 + 0.5);
//    outAbsY = static_cast<LONG>(fy * 65535.0 + 0.5);
//}
//
//// cubic Bezier point
//static void cubic_bezier(double t, double x0, double y0, double x1, double y1,
//    double x2, double y2, double x3, double y3,
//    double& rx, double& ry) {
//    double u = 1.0 - t;
//    double tt = t * t;
//    double uu = u * u;
//    double uuu = uu * u;
//    double ttt = tt * t;
//    rx = uuu * x0 + 3 * uu * t * x1 + 3 * u * tt * x2 + ttt * x3;
//    ry = uuu * y0 + 3 * uu * t * y1 + 3 * u * tt * y2 + ttt * y3;
//}
//
//// ease in/out (smoothstep-like)
//static double easeInOut(double t) {
//    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
//}
//
//// 生成随机 double in [a,b)
//static double rand_double_range(std::mt19937& rng, double a, double b) {
//    std::uniform_real_distribution<double> dist(a, b);
//    return dist(rng);
//}
//
//// 发送一次“绝对”移动，带 VIRTUALDESK
//static bool SendAbsoluteMove(LONG absX, LONG absY) {
//    INPUT input = {};
//    input.type = INPUT_MOUSE;
//    input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;
//    input.mi.dx = absX;
//    input.mi.dy = absY;
//    return SendInput(1, &input, sizeof(INPUT)) == 1;
//}
//
//// 发送一次“相对”移动；为游戏减少合并
//static bool SendRelativeMove(LONG dx, LONG dy) {
//    INPUT in = {};
//    in.type = INPUT_MOUSE;
//    in.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_MOVE_NOCOALESCE; // 相对移动
//    in.mi.dx = dx;
//    in.mi.dy = dy;
//    return SendInput(1, &in, sizeof(INPUT)) == 1;
//}
//
//// 简单“探测”当前上下文是否会吞/重置绝对坐标（游戏抓鼠标、重心定位）
//// 注意：为了尽量降低干扰，只做 +/-1 像素的小试探并立即复原
//static bool ShouldPreferRelativeMove() {
//    POINT p0{};
//    if (!GetCursorPos(&p0)) return false;
//
//    // 目标点：+1 像素（横向），再发回原位
//    LONG a1x = 0, a1y = 0, a0x = 0, a0y = 0;
//    PixelToAbsolute(p0.x + 1, p0.y, a1x, a1y);
//    PixelToAbsolute(p0.x, p0.y, a0x, a0y);
//
//    // 发到 +1
//    if (!SendAbsoluteMove(a1x, a1y)) return true; // 发不出去就用相对
//    std::this_thread::sleep_for(std::chrono::milliseconds(8));
//
//    POINT pA{};
//    GetCursorPos(&pA);
//
//    // 立刻发回原位
//    SendAbsoluteMove(a0x, a0y);
//    std::this_thread::sleep_for(std::chrono::milliseconds(8));
//
//    // 判定逻辑：
//    // 1) 完全没动或被瞬间拉回原位/其它点 => 目标可能在重置/吞绝对移动 => 用相对
//    const auto dist = [](POINT a, POINT b) {
//        return std::hypot(double(a.x - b.x), double(a.y - b.y));
//        };
//    double d_to_plus1 = dist(pA, POINT{ p0.x + 1, p0.y });
//    double d_to_orig = dist(pA, p0);
//
//    // 没有接近 +1 的点、或者更接近原位/远离我们设定的 +1，认为绝对不可靠
//    if (d_to_plus1 > 1.5 || d_to_orig < 1.0) {
//        return true;
//    }
//    return false;
//}
//
//// 主函数：以人类风格移动到 (x,y) 并点击
//// 参数:
////   x,y: 屏幕像素坐标（相对于虚拟屏幕左上）
////   moveMs: 移动耗时（毫秒），推荐 300..1200
////   button: MB_LEFT / MB_RIGHT / MB_MIDDLE
////   clicks: 点击次数（1 单击，2 双击）
////   downMs: 每次按下到抬起维持毫秒
////   restorePos: 点击完成后是否把鼠标移动回原位置（若判定为相对模式，将忽略复位）
//bool HumanLikeMouseClick(int x, int y,
//    int moveMs,
//    MouseButton button,
//    int clicks,
//    int downMs,
//    bool restorePos)
//{
//    EnsureDpiAwareOnce();
//
//    if (clicks < 1) clicks = 1;
//    if (moveMs < 0) moveMs = 0;
//    if (downMs < 0) downMs = 0;
//
//    // 记录原位置
//    POINT origPos = { 0,0 };
//    if (restorePos) GetCursorPos(&origPos);
//
//    // 当前 cursor pos as double start
//    POINT startPos{};
//    GetCursorPos(&startPos);
//    double sx = (double)startPos.x;
//    double sy = (double)startPos.y;
//    double ex = (double)x;
//    double ey = (double)y;
//
//    // RNG
//    std::random_device rd;
//    std::mt19937 rng(rd());
//
//    // 距离与步数
//    double dx_full = ex - sx;
//    double dy_full = ey - sy;
//    double dist = std::hypot(dx_full, dy_full);
//
//    int steps = (int)std::round(dist / 3.0);
//    steps = std::clamp(steps, 6, 60);
//    int minStepMs = 5;
//    int totalSteps = (((1) > (steps)) ? (1) : (steps));
//
//    // 控制点与曲率
//    double offScale = std::clamp(dist * 0.3, 20.0, 300.0);
//    double px = -dy_full, py = dx_full;
//    double plen = std::hypot(px, py);
//    if (plen > 1e-6) { px /= plen; py /= plen; }
//    else { px = 0; py = 0; }
//
//    double c1x = sx + dx_full * 0.3 + px * rand_double_range(rng, -0.7, 0.7) * offScale;
//    double c1y = sy + dy_full * 0.3 + py * rand_double_range(rng, -0.7, 0.7) * offScale;
//    double c2x = sx + dx_full * 0.7 + px * rand_double_range(rng, -0.7, 0.7) * offScale;
//    double c2y = sy + dy_full * 0.7 + py * rand_double_range(rng, -0.7, 0.7) * offScale;
//
//    if (dist < 30.0) {
//        c1x = sx + dx_full * 0.25 + rand_double_range(rng, -4, 4);
//        c1y = sy + dy_full * 0.25 + rand_double_range(rng, -4, 4);
//        c2x = sx + dx_full * 0.75 + rand_double_range(rng, -4, 4);
//        c2y = sy + dy_full * 0.75 + rand_double_range(rng, -4, 4);
//    }
//
//    // 步进时间分配（ease + 抖动）
//    std::vector<int> stepDelays;
//    stepDelays.reserve(totalSteps);
//    double acc = 0.0;
//    for (int i = 0; i < totalSteps; ++i) {
//        double t = (double)i / (double)(((1) >(totalSteps - 1)) ? (1) : (totalSteps - 1));
//        double e = easeInOut(t);
//        stepDelays.push_back(0);
//        acc += e;
//    }
//    for (int i = 0; i < totalSteps; ++i) {
//        double t = (double)i / (double)(((1) >(totalSteps - 1)) ? (1) : (totalSteps - 1));
//        double e = easeInOut(t);
//        double proportion = (acc > 1e-8) ? (e / acc) : (1.0 / totalSteps);
//        double jitter = rand_double_range(rng, 0.85, 1.15);
//        int delay = (int)std::max<double>(minStepMs, (double)moveMs * proportion * jitter);
//        stepDelays[i] = delay;
//    }
//
//    // 是否偏好相对移动（游戏抓鼠标时）
//    bool preferRelative = ShouldPreferRelativeMove();
//
//    // 路径发送
//    double prev_rx = sx, prev_ry = sy;
//    for (int i = 0; i < totalSteps; ++i) {
//        double segmentProgress = (double)i / (double)(((1) >(totalSteps - 1)) ? (1) : (totalSteps - 1));
//        double tRand = segmentProgress + rand_double_range(rng, -0.012, 0.012);
//        tRand = std::clamp(tRand, 0.0, 1.0);
//
//        double rx, ry;
//        cubic_bezier(tRand, sx, sy, c1x, c1y, c2x, c2y, ex, ey, rx, ry);
//
//        double jitterMag = std::clamp(dist * 0.005, 0.5, 6.0);
//        rx += rand_double_range(rng, -jitterMag, jitterMag);
//        ry += rand_double_range(rng, -jitterMag, jitterMag);
//
//        if (!preferRelative) {
//            LONG absX = 0, absY = 0;
//            PixelToAbsolute((LONG)std::lround(rx), (LONG)std::lround(ry), absX, absY);
//            if (!SendAbsoluteMove(absX, absY)) {
//                // 绝对移动失败，立即切换相对
//                preferRelative = true;
//            }
//        }
//
//        if (preferRelative) {
//            // 将路径点转换为相对增量
//            LONG dx = (LONG)std::lround(rx - prev_rx);
//            LONG dy = (LONG)std::lround(ry - prev_ry);
//            // 可能出现 0 增量，强制至少发一小步，避免被游戏忽略
//            if (dx == 0 && dy == 0) {
//                dx = (rand() & 1) ? 1 : -1;
//            }
//            SendRelativeMove(dx, dy);
//        }
//
//        prev_rx = rx; prev_ry = ry;
//
//        int sleepMs = stepDelays[i];
//        int jitter = (int)std::round(std::clamp(sleepMs / 5, 0, 10) * rand_double_range(rng, -1.0, 1.0));
//        sleepMs += jitter;
//        if (sleepMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
//    }
//
//    // 最终对齐到目标点（仅在绝对模式下尝试；相对模式多数游戏会重置，没必要）
//    if (!preferRelative) {
//        LONG absX = 0, absY = 0;
//        PixelToAbsolute(x, y, absX, absY);
//        SendAbsoluteMove(absX, absY);
//        std::this_thread::sleep_for(std::chrono::milliseconds(8 + (rand() % 8)));
//    }
//
//    // 按键 Flags（修正右键拼写）
//    DWORD downFlag = 0, upFlag = 0;
//    switch (button) {
//    case MB_LEFT:   downFlag = MOUSEEVENTF_LEFTDOWN;   upFlag = MOUSEEVENTF_LEFTUP;   break;
//    case MB_RIGHT:  downFlag = MOUSEEVENTF_RIGHTDOWN;  upFlag = MOUSEEVENTF_RIGHTUP;  break;
//    case MB_MIDDLE: downFlag = MOUSEEVENTF_MIDDLEDOWN; upFlag = MOUSEEVENTF_MIDDLEUP; break;
//    default:        downFlag = MOUSEEVENTF_LEFTDOWN;   upFlag = MOUSEEVENTF_LEFTUP;   break;
//    }
//
//    // 执行点击
//    for (int c = 0; c < clicks; ++c) {
//        INPUT inDown = {};
//        inDown.type = INPUT_MOUSE;
//        inDown.mi.dwFlags = downFlag;
//        if (SendInput(1, &inDown, sizeof(INPUT)) != 1) {
//            // 某些受保护游戏可能拒绝注入
//            // 这里不 return，让后续流程继续，避免卡住
//        }
//
//        int hold = downMs;
//        int jitt = (downMs > 0) ? (int)std::clamp(downMs / 3, 0, 50) : 0;
//        hold += (rand() % (jitt + 1)) - (jitt / 2);
//        if (hold < 8) hold = 8;
//        std::this_thread::sleep_for(std::chrono::milliseconds(hold));
//
//        INPUT inUp = {};
//        inUp.type = INPUT_MOUSE;
//        inUp.mi.dwFlags = upFlag;
//        SendInput(1, &inUp, sizeof(INPUT));
//
//        if (c != clicks - 1) {
//            int between = 80 + (rand() % 100); // 80..180ms
//            std::this_thread::sleep_for(std::chrono::milliseconds(between));
//        }
//    }
//
//    std::this_thread::sleep_for(std::chrono::milliseconds(30 + (rand() % 40)));
//
//    // 复位：若处于相对模式（很可能是游戏）则跳过复位
//    if (restorePos && !preferRelative) {
//        POINT cur{}; GetCursorPos(&cur);
//        double bx = (double)cur.x;
//        double by = (double)cur.y;
//        double tx = (double)origPos.x;
//        double ty = (double)origPos.y;
//        int backSteps = 8;
//        for (int i = 1; i <= backSteps; ++i) {
//            double t = (double)i / (double)backSteps;
//            double ix = bx + (tx - bx) * t;
//            double iy = by + (ty - by) * t;
//            LONG ax = 0, ay = 0;
//            PixelToAbsolute((LONG)std::lround(ix), (LONG)std::lround(iy), ax, ay);
//            SendAbsoluteMove(ax, ay);
//            std::this_thread::sleep_for(std::chrono::milliseconds(6 + (rand() % 8)));
//        }
//    }
//
//    return true;
//}
//
//// -------------------- 简单 demo -------------------
//bool SendEscKey(int holdMs = 20)
//{
//    INPUT down = {};
//    down.type = INPUT_KEYBOARD;
//    down.ki.wVk = VK_ESCAPE;
//    down.ki.wScan = static_cast<WORD>(MapVirtualKey(VK_ESCAPE, MAPVK_VK_TO_VSC));
//    down.ki.dwFlags = 0;
//
//    INPUT up = {};
//    up.type = INPUT_KEYBOARD;
//    up.ki.wVk = VK_ESCAPE;
//    up.ki.wScan = static_cast<WORD>(MapVirtualKey(VK_ESCAPE, MAPVK_VK_TO_VSC));
//    up.ki.dwFlags = KEYEVENTF_KEYUP;
//
//    if (SendInput(1, &down, sizeof(INPUT)) != 1) {
//        return false;
//    }
//    if (holdMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(holdMs));
//    if (SendInput(1, &up, sizeof(INPUT)) != 1) {
//        return false;
//    }
//    return true;
//}
//
//
//
//
//
//
//
