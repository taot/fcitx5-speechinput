"""文字粘贴模块 - 复制到剪贴板并使用 ydotool 粘贴"""

import subprocess
import time


def paste_text(text: str) -> None:
    """
    复制文字到剪贴板并模拟 Ctrl+V 粘贴
    
    Args:
        text: 要粘贴的文字
    """
    # 复制到 Wayland 剪贴板
    subprocess.run(["wl-copy"], input=text.encode(), check=True)
    
    time.sleep(0.2)
    
    # 释放所有修饰键，避免干扰粘贴操作
    # Key codes: 42=LShift, 54=RShift, 29=LCtrl, 97=RCtrl, 56=LAlt, 100=RAlt
    subprocess.run(["ydotool", "key", "42:0", "54:0"], check=True)   # 释放 Shift
    subprocess.run(["ydotool", "key", "29:0", "97:0"], check=True)   # 释放 Ctrl
    subprocess.run(["ydotool", "key", "56:0", "100:0"], check=True)  # 释放 Alt
    
    # 模拟 Ctrl+V: 29=Ctrl, 47=V, :1=按下, :0=释放
    subprocess.run(["ydotool", "key", "29:1", "47:1", "47:0", "29:0"], check=True)
