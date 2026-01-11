"""TUI device selector for voice input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import argparse
import subprocess
import sys
import tempfile

import pyaudio
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Static


ENV_PATH = Path(__file__).parent / ".env"


@dataclass
class RecordConfig:
    record_seconds: float = 3.0
    record_rate: int = 44100
    record_channels: int = 1
    record_chunk: int = 1024


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float
    is_default: bool


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def read_env_file(path: Path) -> dict[str, str]:
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
        entries[key.strip()] = _strip_quotes(value)
    return entries


def write_env_file(path: Path, updates: dict[str, str]) -> None:
    lines: list[str] = []
    found_keys: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                lines.append(line)
                continue
            key, _ = stripped.split("=", 1)
            key = key.strip()
            if key in updates:
                lines.append(f"{key}={updates[key]}")
                found_keys.add(key)
            else:
                lines.append(line)
    for key, value in updates.items():
        if key not in found_keys:
            lines.append(f"{key}={value}")
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")


def load_config() -> tuple[RecordConfig, int | None]:
    env = read_env_file(ENV_PATH)
    config = RecordConfig()

    if "RECORD_SECONDS" in env:
        try:
            config.record_seconds = float(env["RECORD_SECONDS"])
        except ValueError:
            pass
    if "RECORD_RATE" in env:
        try:
            config.record_rate = int(env["RECORD_RATE"])
        except ValueError:
            pass
    if "RECORD_CHANNELS" in env:
        try:
            config.record_channels = int(env["RECORD_CHANNELS"])
        except ValueError:
            pass
    if "RECORD_CHUNK" in env:
        try:
            config.record_chunk = int(env["RECORD_CHUNK"])
        except ValueError:
            pass

    device_index = None
    if "DEVICE_INDEX" in env:
        try:
            device_index = int(env["DEVICE_INDEX"])
        except ValueError:
            device_index = None

    return config, device_index


def list_input_devices() -> list[DeviceInfo]:
    p = pyaudio.PyAudio()
    devices: list[DeviceInfo] = []
    default_index: int | None = None
    try:
        default_info = p.get_default_input_device_info()
        default_value = default_info.get("index")
        default_index = int(default_value) if default_value is not None else None
    except Exception:
        default_index = None

    for index in range(p.get_device_count()):
        info = p.get_device_info_by_index(index)
        max_channels = int(info.get("maxInputChannels", 0))
        if max_channels <= 0:
            continue
        devices.append(
            DeviceInfo(
                index=index,
                name=str(info.get("name", "Unknown")),
                max_input_channels=max_channels,
                default_sample_rate=float(info.get("defaultSampleRate", 0)),
                is_default=index == default_index,
            )
        )
    p.terminate()
    return devices


def record_and_playback(
    device_index: int,
    record_seconds: float,
    record_rate: int,
    record_channels: int,
    record_chunk: int,
) -> None:
    p = pyaudio.PyAudio()
    try:
        frames: list[bytes] = []
        stream = p.open(
            format=pyaudio.paInt16,
            channels=record_channels,
            rate=record_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=record_chunk,
        )
        try:
            total_chunks = int(record_rate / record_chunk * record_seconds)
            for _ in range(total_chunks):
                frames.append(stream.read(record_chunk, exception_on_overflow=False))
        finally:
            stream.stop_stream()
            stream.close()

        output = p.open(
            format=pyaudio.paInt16,
            channels=record_channels,
            rate=record_rate,
            output=True,
        )
        try:
            for frame in frames:
                output.write(frame)
        finally:
            output.stop_stream()
            output.close()
    finally:
        p.terminate()


class DeviceSelectorApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #main {
        width: 95%;
        height: 90%;
    }
    #device_table {
        height: 1fr;
        margin-top: 1;
        margin-bottom: 1;
    }
    #controls Input {
        width: 1fr;
        margin-right: 1;
    }
    #status {
        height: 3;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("t", "test_device", "Test"),
        ("enter", "save_device", "Save"),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.record_config, self.env_device_index = load_config()
        self.devices = list_input_devices()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="main"):
            yield Static("Voice Input Device Selector", id="title")
            yield DataTable(id="device_table")
            with Horizontal(id="controls"):
                yield Input(
                    value=str(self.record_config.record_seconds),
                    placeholder="Seconds",
                    id="record_seconds",
                )
                yield Input(
                    value=str(self.record_config.record_rate),
                    placeholder="Rate",
                    id="record_rate",
                )
                yield Input(
                    value=str(self.record_config.record_channels),
                    placeholder="Channels",
                    id="record_channels",
                )
                yield Input(
                    value=str(self.record_config.record_chunk),
                    placeholder="Chunk",
                    id="record_chunk",
                )
            yield Static("Ready", id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Index", "Name", "Channels", "SampleRate", "Default")
        for device in self.devices:
            table.add_row(
                str(device.index),
                device.name,
                str(device.max_input_channels),
                f"{device.default_sample_rate:.0f}",
                "Yes" if device.is_default else "",
                key=str(device.index),
            )
        table.cursor_type = "row"
        if self.devices:
            default_row = 0
            if self.env_device_index is not None:
                for row_index, device in enumerate(self.devices):
                    if device.index == self.env_device_index:
                        default_row = row_index
                        break
            table.cursor_coordinate = (0, default_row)
        else:
            self.set_status("No input devices found")

    def set_status(self, message: str) -> None:
        status = self.query_one("#status", Static)
        status.update(message)

    def get_selected_device_index(self) -> int | None:
        table = self.query_one(DataTable)
        if table.row_count == 0:
            return None
        if table.cursor_row is None:
            return None
        if 0 <= table.cursor_row < len(self.devices):
            return self.devices[table.cursor_row].index
        return None

    def parse_record_config(self) -> RecordConfig | None:
        def parse_float(value: str) -> float:
            return float(value.strip())

        def parse_int(value: str) -> int:
            return int(value.strip())

        try:
            return RecordConfig(
                record_seconds=parse_float(
                    self.query_one("#record_seconds", Input).value
                ),
                record_rate=parse_int(self.query_one("#record_rate", Input).value),
                record_channels=parse_int(
                    self.query_one("#record_channels", Input).value
                ),
                record_chunk=parse_int(self.query_one("#record_chunk", Input).value),
            )
        except ValueError:
            self.set_status("Invalid record parameters")
            return None

    def action_test_device(self) -> None:
        device_index = self.get_selected_device_index()
        if device_index is None:
            self.set_status("No device selected")
            return
        config = self.parse_record_config()
        if config is None:
            return
        self.set_status(f"Recording {config.record_seconds:.1f} seconds...")
        self.run_worker(
            lambda: self._test_device(device_index, config),
            thread=True,
            exclusive=True,
        )

    def _test_device(self, device_index: int, config: RecordConfig) -> None:
        # Run the audio test in a subprocess so any PortAudio/ALSA output or
        # crashes don't break Textual's terminal state.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            prefix="voice-input-device-test-",
            suffix=".log",
        ) as log_file:
            log_path = Path(log_file.name)

        cmd = [
            sys.executable,
            str(Path(__file__)),
            "--test",
            "--device-index",
            str(device_index),
            "--record-seconds",
            str(config.record_seconds),
            "--record-rate",
            str(config.record_rate),
            "--record-channels",
            str(config.record_channels),
            "--record-chunk",
            str(config.record_chunk),
        ]

        try:
            with log_path.open("ab") as log:
                result = subprocess.run(cmd, stdout=log, stderr=log, check=False)
        except Exception as exc:
            self.call_from_thread(self.set_status, f"Test failed: {exc}")
            return

        if result.returncode != 0:
            self.call_from_thread(
                self.set_status,
                f"Test failed (rc={result.returncode}), log: {log_path}",
            )
            return

        self.call_from_thread(self.set_status, "Playback complete")

    def action_save_device(self) -> None:
        device_index = self.get_selected_device_index()
        if device_index is None:
            self.set_status("No device selected")
            return
        config = self.parse_record_config()
        if config is None:
            return
        updates = {
            "DEVICE_INDEX": str(device_index),
            "RECORD_SECONDS": str(config.record_seconds),
            "RECORD_RATE": str(config.record_rate),
            "RECORD_CHANNELS": str(config.record_channels),
            "RECORD_CHUNK": str(config.record_chunk),
        }
        write_env_file(ENV_PATH, updates)
        self.set_status(f"Saved DEVICE_INDEX={device_index}")
        self.exit()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--test", action="store_true", help="Run record/playback test and exit"
    )
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--record-seconds", type=float, default=3.0)
    parser.add_argument("--record-rate", type=int, default=44100)
    parser.add_argument("--record-channels", type=int, default=1)
    parser.add_argument("--record-chunk", type=int, default=1024)
    args = parser.parse_args()

    if args.test:
        if args.device_index is None:
            print("--device-index is required with --test")
            return 2
        record_and_playback(
            device_index=args.device_index,
            record_seconds=args.record_seconds,
            record_rate=args.record_rate,
            record_channels=args.record_channels,
            record_chunk=args.record_chunk,
        )
        return 0

    DeviceSelectorApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
