"""语音转文字模块 - 使用 OpenAI Whisper API

为了减少“无语音也输出一段固定文案”的误触发：
1) 先对录音文件做一次本地能量门限判断（明显静音直接跳过转写）
2) 使用 verbose_json 取回 no_speech_prob 再做二次过滤
"""

from pathlib import Path
import wave

import numpy as np
from openai import OpenAI


# 更偏向“少误触发”：门限设得保守一些。
_SILENCE_MEAN_ABS_THRESHOLD = 350
_ACTIVE_SAMPLE_THRESHOLD = 900
_MIN_ACTIVE_RATIO = 0.005

# Whisper segment 过滤（越严格越少误触发，但可能漏掉很小声）。
_MAX_WEIGHTED_NO_SPEECH_PROB = 0.70
_MIN_SPEECHY_DURATION_SEC = 0.5


def _wav_likely_has_speech(audio_path: Path) -> bool:
    try:
        with wave.open(str(audio_path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
    except Exception:
        # 读失败就交给模型（避免因为格式问题直接丢掉）。
        return True

    if sampwidth != 2 or not frames:
        return True

    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples[::channels]

    mean_abs = float(np.abs(samples.astype(np.int32)).mean())
    if mean_abs < _SILENCE_MEAN_ABS_THRESHOLD:
        return False

    active_ratio = float(
        (np.abs(samples.astype(np.int32)) > _ACTIVE_SAMPLE_THRESHOLD).mean()
    )
    return active_ratio >= _MIN_ACTIVE_RATIO


def _transcription_likely_has_speech(transcription) -> bool:
    segments = getattr(transcription, "segments", None)
    if not segments:
        # 没有细分信息时，至少确保文本非空。
        text = getattr(transcription, "text", "")
        return bool(str(text).strip())

    total = 0.0
    weighted_no_speech = 0.0
    speechy_duration = 0.0

    for seg in segments:
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", 0.0))
        dur = max(0.0, end - start)
        prob = float(getattr(seg, "no_speech_prob", 0.0))

        total += dur
        weighted_no_speech += dur * prob
        if prob < 0.6:
            speechy_duration += dur

    if total <= 0.0:
        return False

    if speechy_duration < _MIN_SPEECHY_DURATION_SEC:
        return False

    return (weighted_no_speech / total) <= _MAX_WEIGHTED_NO_SPEECH_PROB


def transcribe(audio_path: Path) -> str:
    """使用 OpenAI Whisper 将音频转换为文字。"""

    if not _wav_likely_has_speech(audio_path):
        return ""

    client = OpenAI()  # 使用 OPENAI_API_KEY 环境变量

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="zh",
            response_format="verbose_json",
            temperature=0,
        )

    if isinstance(transcription, str):
        return transcription.strip()

    if not _transcription_likely_has_speech(transcription):
        return ""

    return str(getattr(transcription, "text", "")).strip()
