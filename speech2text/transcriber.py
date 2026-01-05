"""语音转文字模块 - 使用 OpenAI Whisper API"""

from pathlib import Path
from openai import OpenAI


def transcribe(audio_path: Path) -> str:
    """
    使用 OpenAI Whisper 将音频转换为文字
    
    Args:
        audio_path: 音频文件路径
    
    Returns:
        转录的文字内容
    """
    client = OpenAI()  # 使用 OPENAI_API_KEY 环境变量
    
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="zh",
            prompt="以下是简体中文内容："
        )
    return transcription.text
