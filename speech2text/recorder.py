"""录音模块 - 支持信号停止、静音检测和超时。

录音参数可通过初始化参数覆盖（用于匹配具体设备）。
"""

import signal
import wave
from pathlib import Path

import numpy as np
import pyaudio


class AudioRecorder:
    """录音器，支持三种停止条件：

    1. 接收到 SIGUSR1 信号
    2. 连续静音超过 SILENCE_DURATION 秒
    3. 录音时长超过 MAX_DURATION 秒
    """

    FORMAT = pyaudio.paInt16

    # 停止条件参数
    SILENCE_THRESHOLD = 500  # 静音阈值 (int16 幅度，范围 0-32767)
    SILENCE_DURATION = 3.0  # 连续静音时长 (秒)
    MAX_DURATION = 60.0  # 最大录音时长 (秒)

    def __init__(
        self,
        *,
        device_index: int | None = None,
        rate: int = 44100,
        channels: int = 1,
        chunk: int = 1024,
    ):
        self.device_index = device_index
        self.rate = rate
        self.channels = channels
        self.chunk = chunk

        self._stop_requested = False
        self._original_handler = None

    def _install_signal_handler(self) -> None:
        self._original_handler = signal.signal(signal.SIGUSR1, self._handle_stop_signal)

    def _restore_signal_handler(self) -> None:
        if self._original_handler is not None:
            signal.signal(signal.SIGUSR1, self._original_handler)

    def _handle_stop_signal(self, signum, frame) -> None:
        self._stop_requested = True

    def _is_silence(self, audio_chunk: bytes) -> bool:
        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        return np.abs(samples).mean() < self.SILENCE_THRESHOLD

    def record(
        self,
        output_path: Path,
        *,
        progress_callback=None,
        stop_on_silence: bool = True,
    ) -> tuple[Path, str]:
        """开始录音。

        Returns:
            (录音文件路径, 停止原因)
            停止原因: "signal" | "silence" | "timeout"
        """

        self._stop_requested = False
        self._install_signal_handler()

        p = pyaudio.PyAudio()
        frames: list[bytes] = []
        stop_reason = "timeout"

        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk,
            )

            silence_chunks = 0
            silence_chunks_threshold = 0
            if stop_on_silence:
                silence_chunks_threshold = int(
                    self.SILENCE_DURATION * self.rate / self.chunk
                )

            max_chunks = int(self.MAX_DURATION * self.rate / self.chunk)

            for i in range(max_chunks):
                if self._stop_requested:
                    stop_reason = "signal"
                    break

                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                except Exception:
                    break

                if stop_on_silence:
                    if self._is_silence(data):
                        silence_chunks += 1
                        if silence_chunks >= silence_chunks_threshold:
                            stop_reason = "silence"
                            break
                    else:
                        silence_chunks = 0

                if progress_callback:
                    elapsed = (i + 1) * self.chunk / self.rate
                    progress_callback(elapsed, f"{elapsed:.1f}s")

            stream.stop_stream()
            stream.close()

            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.FORMAT))
                wf.setframerate(self.rate)
                wf.writeframes(b"".join(frames))

            return output_path, stop_reason

        finally:
            p.terminate()
            self._restore_signal_handler()
