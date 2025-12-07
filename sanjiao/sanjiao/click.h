#pragma once



enum MouseButton {
    MB_LEFT,
    MB_RIGHTT,
    MB_MIDDLE
};
// 主函数：以人类风格移动到 (x,y) 并点击
    // 参数:
    //   x,y: 屏幕像素坐标（相对于虚拟屏幕左上）
    //   moveMs: 移动耗时（毫秒），推荐 300..1200
    //   button: MB_LEFT / MB_RIGHT / MB_MIDDLE
    //   clicks: 点击次数（1 单击，2 双击）
    //   downMs: 每次按下到抬起维持毫秒
    //   restorePos: 点击完成后是否把鼠标移动回原位置
    // 返回: true 表示成功发送了事件（注意：不能保证目标程序处理）
// 建议程序启动时调用一次：开启高 DPI 感知 + 提升计时精度
static void EnsureDpiAwareOnce() {
    static std::once_flag once;
    std::call_once(once, [] {
        // Windows 10+ 优先 Per-Monitor V2，降级到系统 DPI 感知
        HMODULE user32 = GetModuleHandleW(L"user32.dll");
        using SetProcCtxFn = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
        auto pSetCtx = reinterpret_cast<SetProcCtxFn>(GetProcAddress(user32, "SetProcessDpiAwarenessContext"));
        if (pSetCtx) {
            pSetCtx(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
        else {
            SetProcessDPIAware();
        }
        // 提升计时精度到 1ms，便于实现 20ms 级别的总移动时间
        timeBeginPeriod(1);
        });
}
bool HumanLikeMouseClick(int x, int y,
    int moveMs = 13,
    MouseButton button = MB_LEFT,
    int clicks = 1,
    int downMs = 23,
    bool restorePos = false);
bool SendEscKey(int holdMs);