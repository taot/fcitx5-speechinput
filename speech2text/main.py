"""语音输入主程序

功能：
1. 录音（支持信号停止、静音检测、超时）
2. 使用 OpenAI Whisper 转文字
3. 通过 dbus 将文字发送到 fcitx5

停止录音：
- 保持3秒静音
- 达到60秒最大时长
- 发送 SIGUSR1 信号: pkill -SIGUSR1 -f "python.*main.py"
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from notifier import (
    notify_status,
    reset_notification,
    start_notification_thread,
    stop_notification_thread,
)
from recorder import AudioRecorder
from temp_manager import TempFileManager
from transcriber import transcribe


# 从 .env 文件加载环境变量
load_dotenv(Path(__file__).parent / ".env")


def _safe_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _pick_pyaudio_pulse_input_device() -> int | None:
    """Return PyAudio device index for PortAudio Pulse backend."""
    try:
        import pyaudio
    except Exception:
        return None

    p = pyaudio.PyAudio()
    try:
        for index in range(p.get_device_count()):
            info = p.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            name = str(info.get("name", "")).lower()
            if "pulse" in name:
                return int(info.get("index", index))
        return None
    finally:
        p.terminate()


def send_text_via_dbus(text: str) -> None:
    """通过 dbus 将文字发送到 fcitx5"""
    subprocess.run(
        [
            "qdbus",
            "org.fcitx.Fcitx5.SpeechBridge",
            "/org/fcitx/Fcitx5/SpeechBridge",
            "org.fcitx.Fcitx5.SpeechBridge1.SendText",
            text,
        ],
        check=True,
    )


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--record-rate", type=int, default=None)
    parser.add_argument("--record-channels", type=int, default=None)
    parser.add_argument("--record-chunk", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = build_args()

    env_device_index = _safe_int(os.getenv("DEVICE_INDEX"))
    env_record_rate = _safe_int(os.getenv("RECORD_RATE"))
    env_record_channels = _safe_int(os.getenv("RECORD_CHANNELS"))
    env_record_chunk = _safe_int(os.getenv("RECORD_CHUNK"))
    env_pulse_source = os.getenv("PULSE_SOURCE")

    device_index = (
        args.device_index if args.device_index is not None else env_device_index
    )

    if device_index is None and env_pulse_source:
        pulse_device = _pick_pyaudio_pulse_input_device()
        if pulse_device is not None:
            device_index = pulse_device
    record_rate = args.record_rate if args.record_rate is not None else env_record_rate
    record_channels = (
        args.record_channels
        if args.record_channels is not None
        else env_record_channels
    )
    record_chunk = (
        args.record_chunk if args.record_chunk is not None else env_record_chunk
    )

    # Defaults when not configured.
    rate = record_rate or 44100
    channels = record_channels or 1
    chunk = record_chunk or 1024

    reset_notification()
    start_notification_thread()

    try:
        notify_status("🎤 语音输入", "正在初始化...")
        temp_manager = TempFileManager()
        output_path = temp_manager.get_new_file_path()

        notify_status("🎤 语音输入", "正在录音...\n3秒静音或1分钟后自动停止")

        recorder = AudioRecorder(
            device_index=device_index,
            rate=rate,
            channels=channels,
            chunk=chunk,
        )

        def on_progress(elapsed: float, status: str):
            notify_status("🎤 语音输入", f"正在录音... {status}")

        audio_file, stop_reason = recorder.record(
            output_path, progress_callback=on_progress
        )

        stop_reason_text = {
            "signal": "收到停止信号",
            "silence": "检测到静音",
            "timeout": "达到最大时长",
        }.get(stop_reason, stop_reason)

        notify_status("🎤 语音输入", f"录音完成 ({stop_reason_text})\n正在转换文字...")

        temp_manager.cleanup_old_files()

        text = transcribe(audio_file)

        if not text.strip():
            notify_status("🎤 语音输入", "未识别到语音内容", urgency="low")
            return

        preview = text[:80] + ("..." if len(text) > 80 else "")
        notify_status("🎤 语音输入", f"输入: {preview}")
        send_text_via_dbus(text)

        notify_status("🎤 语音输入", f"✓ 完成: {preview}", urgency="low")

    except Exception as e:
        notify_status("🎤 语音输入", f"❌ 错误: {e}", urgency="critical")
        raise

    finally:
        stop_notification_thread()


if __name__ == "__main__":
    main()
