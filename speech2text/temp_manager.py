"""临时文件管理模块 - 保留最近10次录音"""

import time
from pathlib import Path
import tempfile


class TempFileManager:
    MAX_FILES = 10
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "voice-input"
        self.temp_dir.mkdir(exist_ok=True)
    
    def get_new_file_path(self) -> Path:
        """生成新的录音文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return self.temp_dir / f"recording_{timestamp}.wav"
    
    def cleanup_old_files(self) -> None:
        """保留最近 MAX_FILES 个录音文件，删除旧文件"""
        files = sorted(
            self.temp_dir.glob("recording_*.wav"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        for old_file in files[self.MAX_FILES:]:
            try:
                old_file.unlink()
            except OSError:
                pass  # 忽略删除失败的文件
