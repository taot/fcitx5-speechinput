"""语音输入主程序

功能：
1. 录音（支持信号停止、超时）
2. 使用 OpenAI Whisper 转文字
3. 通过 dbus 将文字发送到 fcitx5

快捷键/停止：
- 默认启用 toggle：如果已有实例在运行，再启动会发送 SIGUSR1 请求停止/取消
- 录音阶段收到 SIGUSR1：结束录音并继续转写
- 转写/发送阶段收到 SIGUSR1：取消整次流程并退出
- 录音最长 60 秒兜底超时
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import fcntl

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


def _runtime_dir() -> Path:
    xdg_runtime_dir = os.getenv("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        return Path(xdg_runtime_dir)
    return Path(tempfile.gettempdir())


def _pidfile_path() -> Path:
    return _runtime_dir() / "speech2text.pid"


def _lockfile_path() -> Path:
    return _runtime_dir() / "speech2text.lock"


def _read_pidfile(pidfile: Path) -> int | None:
    try:
        text = pidfile.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _is_our_process(pid: int) -> bool:
    if not _is_process_alive(pid):
        return False

    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        cmdline = cmdline_path.read_bytes()
    except OSError:
        return True

    script_path = str(Path(__file__).resolve()).encode("utf-8")
    return script_path in cmdline


def _write_pidfile(pidfile: Path, pid: int) -> None:
    pidfile.write_text(f"{pid}\n", encoding="utf-8")


def _cleanup_pidfile(pidfile: Path, pid: int) -> None:
    existing_pid = _read_pidfile(pidfile)
    if existing_pid != pid:
        return
    try:
        pidfile.unlink()
    except OSError:
        pass


def _maybe_toggle_or_exit(*, no_toggle: bool) -> None:
    if no_toggle:
        return

    pidfile = _pidfile_path()
    lockfile = _lockfile_path()

    lockfile.parent.mkdir(parents=True, exist_ok=True)

    with open(lockfile, "w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp, fcntl.LOCK_EX)

        existing_pid = _read_pidfile(pidfile)
        if existing_pid is not None and _is_our_process(existing_pid):
            os.kill(existing_pid, signal.SIGUSR1)
            raise SystemExit(0)

        _write_pidfile(pidfile, os.getpid())

    atexit.register(_cleanup_pidfile, pidfile, os.getpid())


def _install_cancel_handler() -> None:
    def _handle_cancel(signum, frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGUSR1, _handle_cancel)


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
            "org.fcitx.Fcitx5.DBusBridge",
            "/org/fcitx/Fcitx5/DBusBridge",
            "org.fcitx.Fcitx5.DBusBridge1.SendText",
            text,
        ],
        check=True,
    )


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--no-toggle",
        action="store_true",
        help="Disable toggle mode; always start a new recording session.",
    )
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--record-rate", type=int, default=None)
    parser.add_argument("--record-channels", type=int, default=None)
    parser.add_argument("--record-chunk", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = build_args()

    _maybe_toggle_or_exit(no_toggle=args.no_toggle)
    _install_cancel_handler()

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

        notify_status("🎤 语音输入", "正在录音...\n再按一次快捷键结束（最长60秒）")

        recorder = AudioRecorder(
            device_index=device_index,
            rate=rate,
            channels=channels,
            chunk=chunk,
        )

        def on_progress(elapsed: float, status: str):
            notify_status("🎤 语音输入", f"正在录音... {status}")

        audio_file, stop_reason = recorder.record(
            output_path, progress_callback=on_progress, stop_on_silence=False
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

    except KeyboardInterrupt:
        notify_status("🎤 语音输入", "已取消", urgency="low")
        return

    except Exception as e:
        notify_status("🎤 语音输入", f"❌ 错误: {e}", urgency="critical")
        raise

    finally:
        stop_notification_thread()


if __name__ == "__main__":
    main()
