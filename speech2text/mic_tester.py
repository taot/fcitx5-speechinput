#!/usr/bin/env python3
"""
Microphone Tester - A simple tool to test and adjust microphone input levels.
Designed for Linux systems with PipeWire/PulseAudio.
"""

import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import sounddevice as sd
import pulsectl

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QMessageBox,
)

from PySide6.QtGui import QPainter, QColor


class AudioWorker(QObject):
    """Handles audio input processing and emits level signals."""

    level_changed = Signal(float)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.stream = None
        self.current_device = None
        self.gain = 1.0
        self.pulse_source_name: str | None = None

    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio status: {status}")
        # Calculate RMS level
        rms = np.sqrt(np.mean(indata**2))
        # Apply gain and convert to 0-1 range (assuming max RMS ~0.5 for loud input)
        level = min(1.0, rms * self.gain * 4)
        self.level_changed.emit(level)

    def start_stream(self, device_index: int, *, pulse_source_name: str | None = None):
        """Start audio input stream for the specified device."""
        self.stop_stream()

        if pulse_source_name is not None:
            self.pulse_source_name = pulse_source_name

        try:
            # When using PortAudio's PulseAudio backend, this controls
            # which Pulse source the stream captures from.
            if self.pulse_source_name:
                os.environ["PULSE_SOURCE"] = self.pulse_source_name

            self.stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=44100,
                blocksize=1024,
                callback=self.audio_callback,
            )
            self.stream.start()
            self.current_device = device_index
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop_stream(self):
        """Stop the current audio stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def set_gain(self, gain: float):
        """Set the display gain multiplier."""
        self.gain = gain


class LEDMeter(QWidget):
    """A segmented LED-style volume meter."""

    def __init__(self, segments: int = 20, parent=None):
        super().__init__(parent)
        self.segments = segments
        self.level = 0.0
        self.setMinimumHeight(20)
        self.setMinimumWidth(200)

    def set_level(self, level: float):
        """Set the current level (0.0 to 1.0)."""
        self.level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event):
        """Draw the LED segments."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        segment_width = (width - (self.segments - 1) * 2) / self.segments
        active_segments = int(self.level * self.segments)

        for i in range(self.segments):
            x = i * (segment_width + 2)

            # Color gradient: green -> yellow -> red
            if i < self.segments * 0.6:
                active_color = QColor(76, 175, 80)  # Green
                inactive_color = QColor(30, 70, 32)
            elif i < self.segments * 0.8:
                active_color = QColor(255, 193, 7)  # Yellow
                inactive_color = QColor(100, 76, 3)
            else:
                active_color = QColor(244, 67, 54)  # Red
                inactive_color = QColor(97, 27, 22)

            if i < active_segments:
                painter.setBrush(active_color)
            else:
                painter.setBrush(inactive_color)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(int(x), 0, int(segment_width), height, 2, 2)


class MicrophoneTester(QWidget):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.pulse: pulsectl.Pulse | None = None
        self.audio_worker = AudioWorker()
        self.capture_device_index: int | None = None
        self.source_list = []  # List of (pulse_source, sd_device_index)
        self.saved_pulse_source_name: str | None = None
        self.env_pulse_source_present = False

        self.init_pulse()
        self.init_ui()
        self.connect_signals()
        self.load_devices()

    def init_pulse(self):
        """Initialize PulseAudio connection."""
        try:
            self.pulse = pulsectl.Pulse("mic-tester")
        except Exception as e:
            self.show_error_and_exit(f"无法连接到 PulseAudio/PipeWire:\n{e}")

    def init_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Microphone Tester")
        # Increase window width by 20% (400 -> 480)
        self.setFixedWidth(480)
        # Disable maximize button
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Microphone selector
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumHeight(32)
        layout.addWidget(self.mic_combo)

        # Test microphone button (disabled)
        self.test_button = QPushButton("Test microphone")
        self.test_button.setEnabled(False)
        self.test_button.setCheckable(True)
        layout.addWidget(self.test_button)

        # LED meter
        self.led_meter = LEDMeter(segments=25)
        self.led_meter.setMinimumHeight(24)
        layout.addWidget(self.led_meter)

        # Save button (persist selection to .env for main.py)
        self.save_button = QPushButton("Save")
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        # Decay timer for LED meter
        self.decay_timer = QTimer()
        self.decay_timer.setInterval(50)
        self.decay_timer.timeout.connect(self.decay_level)
        self.current_level = 0.0

    def find_capture_device_index(self) -> int:
        """Pick a PortAudio input device suitable for Pulse/PipeWire."""
        devices = sd.query_devices()

        # Prefer PulseAudio backend when available.
        for i, dev in enumerate(devices):
            dev_info = cast(dict[str, Any], dev)
            if dev_info.get("max_input_channels", 0) <= 0:
                continue
            name = str(dev_info.get("name", "")).lower()
            if name == "pulse" or " pulse" in name or name.startswith("pulse"):
                return i

        # PipeWire backend sometimes shows up as "pipewire".
        for i, dev in enumerate(devices):
            dev_info = cast(dict[str, Any], dev)
            if dev_info.get("max_input_channels", 0) <= 0:
                continue
            name = str(dev_info.get("name", "")).lower()
            if "pipewire" in name:
                return i

        default_input = sd.default.device[0]
        if default_input is None:
            return 0
        return int(default_input)

    def connect_signals(self):
        """Connect signals and slots."""
        self.mic_combo.currentIndexChanged.connect(self.on_mic_changed)
        self.save_button.clicked.connect(self.on_save_clicked)
        self.audio_worker.level_changed.connect(self.on_level_changed)
        self.audio_worker.error_occurred.connect(self.on_audio_error)

    def get_saved_pulse_source_name(self) -> str | None:
        env_path = Path(__file__).parent / ".env"
        try:
            return self.read_env_file(env_path).get("PULSE_SOURCE") or None
        except Exception:
            return None

    def get_current_pulse_source_name(self) -> str | None:
        index = self.mic_combo.currentIndex()
        if index < 0 or index >= len(self.source_list):
            return None
        source, _ = self.source_list[index]
        return source.name

    def has_unsaved_changes(self) -> bool:
        current = self.get_current_pulse_source_name()
        if not self.env_pulse_source_present:
            return bool(current)
        return current != self.saved_pulse_source_name

    def update_save_button_state(self) -> None:
        self.save_button.setEnabled(self.has_unsaved_changes())

    def save_current_selection(self) -> bool:
        source_name = self.get_current_pulse_source_name()
        if not source_name:
            QMessageBox.warning(self, "提示", "请先选择一个麦克风")
            return False

        env_path = Path(__file__).parent / ".env"
        try:
            self.write_env_file(env_path, {"PULSE_SOURCE": source_name})
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")
            return False

        self.saved_pulse_source_name = source_name
        self.env_pulse_source_present = True
        self.update_save_button_state()
        return True

    def load_devices(self):
        """Load available microphone devices."""
        self.source_list.clear()
        self.mic_combo.clear()

        try:
            pulse = self.pulse
            if pulse is None:
                self.show_error_and_exit("PulseAudio/PipeWire 未初始化")

            # Get PulseAudio sources
            sources = pulse.source_list()

            # Use a single PortAudio backend device for capturing, and select
            # the actual microphone via PULSE_SOURCE when possible.
            self.capture_device_index = self.find_capture_device_index()

            for source in sources:
                # Skip monitor sources (they capture output, not input)
                if "monitor" in source.name.lower():
                    continue

                self.source_list.append((source, self.capture_device_index))
                self.mic_combo.addItem(source.description)

            if not self.source_list:
                self.show_error_and_exit("未找到可用的麦克风设备")

            saved_source = self.get_saved_pulse_source_name()
            self.saved_pulse_source_name = saved_source
            self.env_pulse_source_present = saved_source is not None
            if saved_source:
                for i, (source, _) in enumerate(self.source_list):
                    if source.name == saved_source:
                        self.mic_combo.setCurrentIndex(i)
                        break

            self.update_save_button_state()

        except Exception as e:
            self.show_error_and_exit(f"加载设备列表失败:\n{e}")

    def read_env_file(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        entries: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            entries[key.strip()] = value.strip().strip('"').strip("'")
        return entries

    def write_env_file(self, path: Path, updates: dict[str, str]) -> None:
        existing = self.read_env_file(path)
        existing.update(updates)

        lines: list[str] = []
        for key, value in existing.items():
            lines.append(f"{key}={value}")

        # Keep file ending with newline.
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def on_save_clicked(self) -> None:
        if self.save_current_selection():
            QMessageBox.information(self, "提示", "已保存到 .env (PULSE_SOURCE)")

    def on_mic_changed(self, index: int):
        """Handle microphone selection change."""
        if index < 0 or index >= len(self.source_list):
            return

        source, sd_index = self.source_list[index]

        # Start audio stream. Prefer capturing via Pulse backend and bind to the
        # selected microphone using PULSE_SOURCE.
        self.audio_worker.start_stream(sd_index, pulse_source_name=source.name)
        self.decay_timer.start()
        self.update_save_button_state()

    def on_level_changed(self, level: float):
        """Handle audio level update from worker."""
        # Use immediate update for rising levels, decay for falling
        if level > self.current_level:
            self.current_level = level
        self.led_meter.set_level(self.current_level)

    def decay_level(self):
        """Gradually decay the level display."""
        self.current_level *= 0.85
        if self.current_level < 0.01:
            self.current_level = 0
        self.led_meter.set_level(self.current_level)

    def on_audio_error(self, error_msg: str):
        """Handle audio errors."""
        self.show_error_and_exit(f"音频错误:\n{error_msg}")

    def show_error_and_exit(self, message: str):
        """Show error dialog and exit application."""
        QMessageBox.critical(self, "错误", message)
        sys.exit(1)

    def closeEvent(self, event):
        """Clean up on window close."""
        if self.has_unsaved_changes():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("未保存")
            box.setText("当前麦克风选择尚未保存到 .env")
            box.setInformativeText("是否保存后再退出？")

            save_btn = box.addButton("保存并退出", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = box.addButton(
                "不保存退出", QMessageBox.ButtonRole.DestructiveRole
            )
            box.addButton(QMessageBox.StandardButton.Cancel)

            box.exec()
            clicked = box.clickedButton()

            if clicked == save_btn:
                if not self.save_current_selection():
                    event.ignore()
                    return
            elif clicked != discard_btn:
                event.ignore()
                return

        self.decay_timer.stop()
        self.audio_worker.stop_stream()
        if self.pulse:
            self.pulse.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MicrophoneTester()
    window.show()

    # Trigger initial device selection (honor .env selection).
    if window.mic_combo.count() > 0:
        window.on_mic_changed(window.mic_combo.currentIndex())

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
