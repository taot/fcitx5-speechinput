"""桌面通知模块 - 使用 notify-send 显示状态，后台线程保持通知可见"""

import subprocess
import threading
import time


class NotificationManager:
    """
    通知管理器，使用后台线程定期刷新通知以保持可见
    """
    
    REFRESH_INTERVAL = 3.0  # 刷新间隔（秒）
    
    def __init__(self):
        self._notification_id: int | None = None
        self._current_title: str = ""
        self._current_message: str = ""
        self._current_urgency: str = "normal"
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
    
    def _send_notification(self) -> None:
        """发送/更新通知"""
        cmd = [
            "notify-send",
            "-u", self._current_urgency,
        ]
        
        if self._notification_id is None:
            cmd.append("-p")
        else:
            cmd.extend(["-r", str(self._notification_id)])
        
        cmd.extend([self._current_title, self._current_message])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if self._notification_id is None and result.stdout.strip():
                self._notification_id = int(result.stdout.strip())
        except subprocess.CalledProcessError:
            pass  # 忽略通知发送失败
    
    def _refresh_loop(self) -> None:
        """后台线程刷新循环"""
        while self._running:
            with self._lock:
                if self._current_title:
                    self._send_notification()
            time.sleep(self.REFRESH_INTERVAL)
    
    def start(self) -> None:
        """启动后台刷新线程"""
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止后台刷新线程"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def update(self, title: str, message: str, urgency: str = "normal") -> None:
        """
        更新通知内容
        
        Args:
            title: 通知标题
            message: 通知内容
            urgency: 紧急程度 (low/normal/critical)
        """
        with self._lock:
            self._current_title = title
            self._current_message = message
            self._current_urgency = urgency
            self._send_notification()  # 立即发送
    
    def reset(self) -> None:
        """重置通知状态"""
        with self._lock:
            self._notification_id = None
            self._current_title = ""
            self._current_message = ""


# 全局单例
_manager = NotificationManager()


def notify_status(title: str, message: str, urgency: str = "normal") -> None:
    """
    显示/更新桌面通知
    
    Args:
        title: 通知标题
        message: 通知内容
        urgency: 紧急程度 (low/normal/critical)
    """
    _manager.update(title, message, urgency)


def start_notification_thread() -> None:
    """启动通知刷新线程"""
    _manager.start()


def stop_notification_thread() -> None:
    """停止通知刷新线程"""
    _manager.stop()


def reset_notification() -> None:
    """重置通知 ID，下次调用 notify_status 时创建新通知"""
    _manager.reset()
