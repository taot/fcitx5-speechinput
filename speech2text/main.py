"""语音输入主程序

功能：
1. 录音（支持信号停止、静音检测、超时）
2. 使用 OpenAI Whisper 转文字
3. 通过 dbus 将文字发送到 fcitx5

使用方法：
    python main.py [device_index]

停止录音：
    - 保持3秒静音
    - 达到60秒最大时长
    - 发送 SIGUSR1 信号: pkill -SIGUSR1 -f "python.*main.py"
"""

import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv(Path(__file__).parent / ".env")

from notifier import notify_status, reset_notification, start_notification_thread, stop_notification_thread
from recorder import AudioRecorder
from temp_manager import TempFileManager
from transcriber import transcribe


def send_text_via_dbus(text: str) -> None:
    """通过 dbus 将文字发送到 fcitx5"""
    subprocess.run(
        [
            'qdbus',
            'org.fcitx.Fcitx5.SpeechBridge',
            '/org/fcitx/Fcitx5/SpeechBridge',
            'org.fcitx.Fcitx5.SpeechBridge1.SendText',
            text
        ],
        check=True
    )


def main() -> None:
    """主函数"""
    # 解析设备索引参数
    # device_index = int(sys.argv[1]) if len(sys.argv) > 1 else None
    device_index = 11
    
    # 启动通知刷新线程
    reset_notification()
    start_notification_thread()
    
    try:
        # 1. 准备临时文件
        notify_status("🎤 语音输入", "正在初始化...")
        temp_manager = TempFileManager()
        output_path = temp_manager.get_new_file_path()

        # 2. 录音
        notify_status("🎤 语音输入", "正在录音...\n3秒静音或1分钟后自动停止")
        
        recorder = AudioRecorder(device_index=device_index)
        
        def on_progress(elapsed: float, status: str):
            notify_status("🎤 语音输入", f"正在录音... {status}")
        
        audio_file, stop_reason = recorder.record(output_path, progress_callback=on_progress)
        
        stop_reason_text = {
            "signal": "收到停止信号",
            "silence": "检测到静音",
            "timeout": "达到最大时长"
        }.get(stop_reason, stop_reason)
        
        notify_status("🎤 语音输入", f"录音完成 ({stop_reason_text})\n正在转换文字...")

        # 3. 清理旧文件
        temp_manager.cleanup_old_files()

        # 4. 转文字
        text = transcribe(audio_file)
        
        if not text.strip():
            notify_status("🎤 语音输入", "未识别到语音内容", urgency="low")
            return

        # 5. 通过 dbus 发送到 fcitx5
        notify_status("🎤 语音输入", f"输入: {text[:80]}...")
        send_text_via_dbus(text)
        
        notify_status("🎤 语音输入", f"✓ 完成: {text[:80]}...", urgency="low")
        
    except Exception as e:
        notify_status("🎤 语音输入", f"❌ 错误: {e}", urgency="critical")
        raise
    
    finally:
        # 停止通知刷新线程
        stop_notification_thread()


if __name__ == '__main__':
    main()

