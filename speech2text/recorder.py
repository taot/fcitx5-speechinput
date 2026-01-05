"""录音模块 - 支持信号停止、静音检测和超时"""

import signal
import wave
from pathlib import Path

import numpy as np
import pyaudio


class AudioRecorder:
    """
    录音器，支持三种停止条件：
    1. 接收到 SIGUSR1 信号
    2. 连续静音超过 SILENCE_DURATION 秒
    3. 录音时长超过 MAX_DURATION 秒
    """
    
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    # 停止条件参数
    SILENCE_THRESHOLD = 500      # 静音阈值 (int16 幅度，范围 0-32767)
    SILENCE_DURATION = 3.0       # 连续静音时长 (秒)
    MAX_DURATION = 60.0          # 最大录音时长 (秒)
    
    def __init__(self, device_index: int | None = None):
        """
        初始化录音器
        
        Args:
            device_index: 音频设备索引，None 表示使用默认设备
        """
        self.device_index = device_index
        self._stop_requested = False
        self._original_handler = None
    
    def _install_signal_handler(self) -> None:
        """安装 SIGUSR1 信号处理器"""
        self._original_handler = signal.signal(signal.SIGUSR1, self._handle_stop_signal)
    
    def _restore_signal_handler(self) -> None:
        """恢复原始信号处理器"""
        if self._original_handler is not None:
            signal.signal(signal.SIGUSR1, self._original_handler)
    
    def _handle_stop_signal(self, signum, frame) -> None:
        """处理 SIGUSR1 停止信号"""
        self._stop_requested = True
    
    def _is_silence(self, audio_chunk: bytes) -> bool:
        """
        检测音频块是否为静音
        
        Args:
            audio_chunk: 原始音频数据
        
        Returns:
            True 如果音量低于阈值
        """
        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        return np.abs(samples).mean() < self.SILENCE_THRESHOLD
    
    def record(self, output_path: Path, progress_callback=None) -> tuple[Path, str]:
        """
        开始录音
        
        Args:
            output_path: 输出文件路径
            progress_callback: 进度回调函数，接收 (elapsed_seconds, status) 参数
        
        Returns:
            (录音文件路径, 停止原因)
            停止原因: "signal" | "silence" | "timeout"
        """
        self._stop_requested = False
        self._install_signal_handler()
        
        p = pyaudio.PyAudio()
        frames = []
        stop_reason = "timeout"
        
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.CHUNK
            )
            
            silence_chunks = 0
            silence_chunks_threshold = int(self.SILENCE_DURATION * self.RATE / self.CHUNK)
            max_chunks = int(self.MAX_DURATION * self.RATE / self.CHUNK)
            
            for i in range(max_chunks):
                # 检查信号停止
                if self._stop_requested:
                    stop_reason = "signal"
                    break
                
                # 读取音频数据
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception:
                    break
                
                # 检测静音
                if self._is_silence(data):
                    silence_chunks += 1
                    if silence_chunks >= silence_chunks_threshold:
                        stop_reason = "silence"
                        break
                else:
                    silence_chunks = 0
                
                # 进度回调
                if progress_callback:
                    elapsed = (i + 1) * self.CHUNK / self.RATE
                    progress_callback(elapsed, f"{elapsed:.1f}s")
            
            stream.stop_stream()
            stream.close()
            
            # 保存 WAV 文件
            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            
            return output_path, stop_reason
            
        finally:
            p.terminate()
            self._restore_signal_handler()
