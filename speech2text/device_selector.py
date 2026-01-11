"""Voice input device selector.

Two modes:

1) TUI (default):
   - `a`: auto scan all input devices (2s silence + 10/15s speak)
   - after scan, auto-replay top candidate once
   - `y`: accept candidate (enables `s`)
   - `n`: play next candidate
   - `p`: replay current candidate
   - `s`: save accepted candidate to `.env` (stay in TUI)
   - `d`: toggle speak duration 10s/15s (restarts scan if running)

2) Worker/CLI utilities:
   - `--scan-all-worker`: run scan and emit JSONL to stdout
   - `--record-to` / `--play-from`: record/play WAV for debugging

Saved keys in `.env`:
- DEVICE_INDEX
- RECORD_RATE
- RECORD_CHANNELS
- RECORD_CHUNK
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
import pyaudio
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.coordinate import Coordinate
from textual.widgets import DataTable, Footer, Header, Static


ENV_PATH = Path(__file__).parent / ".env"


MONITOR_LIKE_RE = re.compile(
    r"\b(monitor|loopback|virtual|dummy|null)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class AudioParams:
    rate: int
    channels: int
    chunk: int


@dataclass(frozen=True)
class InputDevice:
    index: int
    hostapi: str
    name: str
    max_input_channels: int
    default_sample_rate: int
    is_default: bool
    normalized_name: str
    monitor_like: bool
    dup_rank: int


@dataclass(frozen=True)
class ScanMetrics:
    noise_rms: float
    speech_rms: float
    snr_like: float
    activity: float
    zero_ratio: float
    clip_ratio: float


@dataclass(frozen=True)
class ScanResult:
    device: InputDevice
    status: str  # ok|fail
    params: AudioParams | None
    metrics: ScanMetrics | None
    score: float | None
    penalty_monitor: float
    penalty_dup: float
    err: str | None


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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _jsonl(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _normalize_name(name: str) -> str:
    lowered = name.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _chunk_rms(data: bytes) -> float:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def _analyze_chunks(
    chunks: list[bytes],
    *,
    noise_rms: float,
    activity_multiplier: float = 1.5,
    activity_floor: float = 80.0,
) -> tuple[float, float, float, float]:
    """Return (speech_rms, activity, zero_ratio, clip_ratio) based on chunks."""

    if not chunks:
        return 0.0, 0.0, 1.0, 0.0

    # Per-chunk RMS for activity.
    rms_values = [_chunk_rms(chunk) for chunk in chunks if chunk]
    speech_rms = float(sum(rms_values) / max(1, len(rms_values)))

    threshold = max(activity_floor, noise_rms * activity_multiplier)
    active = sum(1 for v in rms_values if v > threshold)
    activity = active / max(1, len(rms_values))

    samples = np.frombuffer(b"".join(chunks), dtype=np.int16)
    if samples.size == 0:
        return speech_rms, activity, 1.0, 0.0

    zero_ratio = float(np.mean(samples == 0))
    clip_ratio = float(np.mean(np.abs(samples) >= 32700))

    return speech_rms, activity, zero_ratio, clip_ratio


def _open_and_read(
    *,
    device_index: int,
    params: AudioParams,
    seconds: float,
) -> list[bytes]:
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=params.channels,
            rate=params.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=params.chunk,
        )

        chunks: list[bytes] = []
        num_chunks = max(1, int(max(0.05, seconds) * params.rate / params.chunk))
        for _ in range(num_chunks):
            chunks.append(stream.read(params.chunk, exception_on_overflow=False))
        return chunks
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()


def _try_open_stream(device_index: int, params: AudioParams) -> bool:
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=params.channels,
            rate=params.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=params.chunk,
        )
        # Validate read.
        for _ in range(2):
            stream.read(params.chunk, exception_on_overflow=False)
        return True
    except Exception:
        return False
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()


def _candidate_params_for_device(device: InputDevice) -> list[AudioParams]:
    rates: list[int] = []
    if device.default_sample_rate > 0:
        rates.append(device.default_sample_rate)
    for rate in (48000, 44100, 16000):
        if rate not in rates:
            rates.append(rate)

    candidates: list[AudioParams] = []
    for chunk in (1024, 2048, 512):
        for channels in (1, 2):
            if channels > device.max_input_channels:
                continue
            for rate in rates:
                candidates.append(
                    AudioParams(rate=rate, channels=channels, chunk=chunk)
                )

    # De-dup preserving order.
    seen: set[tuple[int, int, int]] = set()
    unique: list[AudioParams] = []
    for params in candidates:
        key = (params.rate, params.channels, params.chunk)
        if key in seen:
            continue
        seen.add(key)
        unique.append(params)

    return unique


def _pick_params_with_fallback(device: InputDevice) -> list[AudioParams]:
    """Pick up to 2 openable param sets.

    We try preferred params first. The second entry (if any) is a fallback that
    often helps when the first one yields silence.
    """

    openable: list[AudioParams] = []
    for params in _candidate_params_for_device(device):
        if _try_open_stream(device.index, params):
            openable.append(params)
            break

    if not openable:
        return []

    first = openable[0]

    # Fallback: toggle channels first, then adjust rate.
    fallbacks: list[AudioParams] = []
    if first.channels == 1 and device.max_input_channels >= 2:
        fallbacks.append(AudioParams(rate=first.rate, channels=2, chunk=first.chunk))
    elif first.channels == 2:
        fallbacks.append(AudioParams(rate=first.rate, channels=1, chunk=first.chunk))

    for rate in (48000, 44100, 16000):
        if rate != first.rate:
            fallbacks.append(
                AudioParams(rate=rate, channels=first.channels, chunk=first.chunk)
            )

    for candidate in fallbacks:
        if _try_open_stream(device.index, candidate):
            openable.append(candidate)
            break

    return openable[:2]


def _compute_penalties(device: InputDevice) -> tuple[float, float]:
    penalty_monitor = 1.0 if device.monitor_like else 0.0
    penalty_dup = 0.3 * max(0, device.dup_rank)
    return penalty_monitor, penalty_dup


def _score_device(
    *,
    metrics: ScanMetrics,
    device: InputDevice,
) -> float:
    penalty_monitor, penalty_dup = _compute_penalties(device)
    default_bonus = 0.4 if device.is_default else 0.0
    score = (
        metrics.snr_like
        + 0.5 * metrics.activity
        - 3.0 * metrics.zero_ratio
        - 3.0 * metrics.clip_ratio
        - penalty_monitor
        - penalty_dup
        + default_bonus
    )
    return float(score)


def _write_wav(path: Path, *, params: AudioParams, frames: list[bytes]) -> None:
    p = pyaudio.PyAudio()
    try:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(params.channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(params.rate)
            wf.writeframes(b"".join(frames))
    finally:
        p.terminate()


def list_input_devices_extended() -> list[InputDevice]:
    p = pyaudio.PyAudio()
    try:
        default_index: int | None = None
        try:
            default_info = p.get_default_input_device_info()
            default_value = default_info.get("index")
            default_index = int(default_value) if default_value is not None else None
        except Exception:
            default_index = None

        raw_devices: list[dict[str, Any]] = []
        for index in range(p.get_device_count()):
            info = p.get_device_info_by_index(index)
            max_channels = int(info.get("maxInputChannels", 0))
            if max_channels <= 0:
                continue
            host_api_index = int(info.get("hostApi", 0))
            host_api_name = str(
                p.get_host_api_info_by_index(host_api_index).get("name", "")
            )
            name = str(info.get("name", "Unknown"))
            raw_devices.append(
                {
                    "index": index,
                    "hostapi": host_api_name,
                    "name": name,
                    "max_input_channels": max_channels,
                    "default_sample_rate": int(float(info.get("defaultSampleRate", 0))),
                    "is_default": index == default_index,
                }
            )

        # Duplicate ranks: by normalized_name (across HostAPI).
        groups: dict[str, list[dict[str, Any]]] = {}
        for d in raw_devices:
            key = _normalize_name(d["name"])
            groups.setdefault(key, []).append(d)

        dup_rank_by_index: dict[int, int] = {}
        for key, group in groups.items():
            # Prefer default first in rank ordering.
            group_sorted = sorted(
                group, key=lambda x: (not x["is_default"], x["hostapi"], x["index"])
            )
            for rank, item in enumerate(group_sorted):
                dup_rank_by_index[item["index"]] = rank

        devices: list[InputDevice] = []
        for d in raw_devices:
            norm = _normalize_name(d["name"])
            devices.append(
                InputDevice(
                    index=d["index"],
                    hostapi=d["hostapi"],
                    name=d["name"],
                    max_input_channels=d["max_input_channels"],
                    default_sample_rate=d["default_sample_rate"],
                    is_default=d["is_default"],
                    normalized_name=norm,
                    monitor_like=bool(MONITOR_LIKE_RE.search(norm)),
                    dup_rank=dup_rank_by_index.get(d["index"], 0),
                )
            )

        return devices
    finally:
        p.terminate()


def run_scan_one_worker(
    *,
    session_id: int,
    device_index: int,
    hostapi: str,
    name: str,
    max_input_channels: int,
    default_sample_rate: int,
    is_default: bool,
    dup_rank: int,
    monitor_like: bool,
    sample_seconds: float,
    allow_fallback: bool,
) -> int:
    """Scan a single device and emit one JSON line.

    This runs in its own process so a PortAudio crash only kills this worker.
    """

    device = InputDevice(
        index=device_index,
        hostapi=hostapi,
        name=name,
        max_input_channels=max_input_channels,
        default_sample_rate=default_sample_rate,
        is_default=is_default,
        normalized_name=_normalize_name(name),
        monitor_like=monitor_like,
        dup_rank=dup_rank,
    )

    penalty_monitor, penalty_dup = _compute_penalties(device)

    candidates = _pick_params_with_fallback(device)
    if not candidates:
        _jsonl(
            {
                "type": "scan_one",
                "session": session_id,
                "status": "fail",
                "device_index": device_index,
                "hostapi": hostapi,
                "name": name,
                "is_default": is_default,
                "err": "no openable params",
            }
        )
        return 3

    def measure(params: AudioParams, seconds: float) -> tuple[ScanMetrics, list[bytes]]:
        chunks = _open_and_read(
            device_index=device_index, params=params, seconds=seconds
        )
        rms_values = [_chunk_rms(chunk) for chunk in chunks if chunk]
        noise_rms = float(np.percentile(rms_values, 20)) if rms_values else 0.0
        speech_rms, activity, zero_ratio, clip_ratio = _analyze_chunks(
            chunks, noise_rms=noise_rms
        )
        snr_like = (speech_rms - noise_rms) / (noise_rms + 1e-6)
        return (
            ScanMetrics(
                noise_rms=noise_rms,
                speech_rms=speech_rms,
                snr_like=float(snr_like),
                activity=float(activity),
                zero_ratio=float(zero_ratio),
                clip_ratio=float(clip_ratio),
            ),
            chunks,
        )

    best_params = candidates[0]
    best_metrics, best_chunks = measure(best_params, sample_seconds)

    if allow_fallback and len(candidates) >= 2:
        looks_silent = best_metrics.zero_ratio > 0.95 or best_metrics.speech_rms <= max(
            best_metrics.noise_rms * 1.2, 60.0
        )
        if looks_silent:
            try:
                fallback_seconds = max(0.12, min(sample_seconds * 0.5, sample_seconds))
                fallback_metrics, fallback_chunks = measure(
                    candidates[1], fallback_seconds
                )
                if fallback_metrics.snr_like > best_metrics.snr_like:
                    best_params = candidates[1]
                    best_metrics = fallback_metrics
                    best_chunks = fallback_chunks
            except Exception:
                pass

    score = _score_device(metrics=best_metrics, device=device)

    wav_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            prefix=f"voice-input-scan-one-{device_index}-",
            suffix=".wav",
        ) as wav_file:
            tmp_path = Path(wav_file.name)
        _write_wav(tmp_path, params=best_params, frames=best_chunks)
        wav_path = str(tmp_path)
    except Exception:
        wav_path = None

    _jsonl(
        {
            "type": "scan_one",
            "session": session_id,
            "status": "ok",
            "device_index": device_index,
            "hostapi": hostapi,
            "name": name,
            "is_default": is_default,
            "rate": best_params.rate,
            "channels": best_params.channels,
            "chunk": best_params.chunk,
            "noise_rms": round(best_metrics.noise_rms, 2),
            "speech_rms": round(best_metrics.speech_rms, 2),
            "snr_like": round(best_metrics.snr_like, 3),
            "activity": round(best_metrics.activity, 3),
            "zero_ratio": round(best_metrics.zero_ratio, 4),
            "clip_ratio": round(best_metrics.clip_ratio, 4),
            "penalty": {"monitor": penalty_monitor, "dup": penalty_dup},
            "score": round(score, 4),
            "wav": wav_path,
        }
    )
    return 0


def run_scan_all_worker(
    *,
    session_id: int,
    silence_seconds: float,
    speak_seconds: float,
    top_k: int,
) -> int:
    """Scan all devices, but isolate each device scan in a subprocess.

    This avoids `rc=-11` killing the whole scan when PortAudio crashes.
    """

    devices = list_input_devices_extended()
    total_devices = len(devices)

    _jsonl({"type": "meta", "session": session_id, "devices": total_devices})
    if total_devices == 0:
        _jsonl({"type": "done", "session": session_id, "ranking": [], "wav_paths": {}})
        return 0

    # Phase 1: silence countdown (user guidance).
    _jsonl(
        {
            "type": "phase",
            "session": session_id,
            "phase": "silence",
            "remaining": silence_seconds,
        }
    )
    silence_end = time.monotonic() + max(0.0, silence_seconds)
    while True:
        remaining = silence_end - time.monotonic()
        if remaining <= 0:
            break
        _jsonl(
            {
                "type": "phase",
                "session": session_id,
                "phase": "silence",
                "remaining": round(max(0.0, remaining), 2),
            }
        )
        time.sleep(min(0.25, remaining))

    # Phase 2: speak + scanning.
    _jsonl(
        {
            "type": "phase",
            "session": session_id,
            "phase": "speak",
            "remaining": speak_seconds,
        }
    )

    # Leave time for process overhead.
    raw_per_device = speak_seconds / max(1, total_devices)
    sample_seconds = min(0.8, max(0.15, raw_per_device * 0.7))

    speak_end = time.monotonic() + max(0.0, speak_seconds)

    results: list[dict[str, Any]] = []

    for idx, device in enumerate(devices, start=1):
        remaining = max(0.0, speak_end - time.monotonic())
        _jsonl(
            {
                "type": "progress",
                "session": session_id,
                "phase": "speak",
                "current": idx,
                "total": total_devices,
                "device_index": device.index,
                "remaining": remaining,
            }
        )

        if time.monotonic() >= speak_end:
            # Out of speak window; stop scanning.
            _jsonl(
                {
                    "type": "device",
                    "session": session_id,
                    "status": "fail",
                    "device_index": device.index,
                    "hostapi": device.hostapi,
                    "name": device.name,
                    "is_default": device.is_default,
                    "err": "speak window ended",
                }
            )
            continue

        cmd = [
            sys.executable,
            str(Path(__file__)),
            "--scan-one-worker",
            "--session-id",
            str(session_id),
            "--device-index",
            str(device.index),
            "--device-name",
            device.name,
            "--hostapi",
            device.hostapi,
            "--max-input-channels",
            str(device.max_input_channels),
            "--default-sample-rate",
            str(device.default_sample_rate),
            "--dup-rank",
            str(device.dup_rank),
            "--sample-seconds",
            str(sample_seconds),
        ]
        if device.is_default:
            cmd.append("--is-default")
        if device.monitor_like:
            cmd.append("--monitor-like")

        try:
            budget = min(2.0, max(0.6, sample_seconds + 0.8))
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=budget,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _jsonl(
                {
                    "type": "device",
                    "session": session_id,
                    "status": "fail",
                    "device_index": device.index,
                    "hostapi": device.hostapi,
                    "name": device.name,
                    "is_default": device.is_default,
                    "err": "timeout",
                }
            )
            continue

        if completed.returncode != 0:
            err = (
                "crash" if completed.returncode == -11 else f"rc={completed.returncode}"
            )
            _jsonl(
                {
                    "type": "device",
                    "session": session_id,
                    "status": "fail",
                    "device_index": device.index,
                    "hostapi": device.hostapi,
                    "name": device.name,
                    "is_default": device.is_default,
                    "err": err,
                }
            )
            continue

        line = (
            (completed.stdout or "").strip().splitlines()[-1]
            if completed.stdout
            else ""
        )
        try:
            payload = json.loads(line)
        except Exception:
            _jsonl(
                {
                    "type": "device",
                    "session": session_id,
                    "status": "fail",
                    "device_index": device.index,
                    "hostapi": device.hostapi,
                    "name": device.name,
                    "is_default": device.is_default,
                    "err": "bad output",
                }
            )
            continue

        if payload.get("status") != "ok":
            _jsonl(
                {
                    "type": "device",
                    "session": session_id,
                    "status": "fail",
                    "device_index": device.index,
                    "hostapi": device.hostapi,
                    "name": device.name,
                    "is_default": device.is_default,
                    "err": payload.get("err") or "scan failed",
                }
            )
            continue

        # Emit the device update as-is (TUI uses it).
        payload["type"] = "device"
        payload["session"] = session_id
        _jsonl(payload)
        results.append(payload)

    ok = [r for r in results if r.get("status") == "ok" and r.get("score") is not None]
    ok.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    ranking = [int(r["device_index"]) for r in ok]

    # Keep only top_k wavs; delete the rest.
    wav_paths: dict[str, str] = {}
    keep = set(ranking[: max(0, top_k)])
    for item in ok:
        idx = int(item["device_index"])
        wav = item.get("wav")
        if not wav:
            continue
        if idx in keep:
            wav_paths[str(idx)] = str(wav)
        else:
            try:
                Path(str(wav)).unlink()
            except OSError:
                pass

    _jsonl(
        {
            "type": "done",
            "session": session_id,
            "ranking": ranking,
            "wav_paths": wav_paths,
        }
    )
    return 0


def _run_scan_all_worker_legacy(
    *,
    session_id: int,
    silence_seconds: float,
    speak_seconds: float,
    top_k: int,
) -> int:
    devices = list_input_devices_extended()
    total_devices = len(devices)

    _jsonl({"type": "meta", "session": session_id, "devices": total_devices})

    if total_devices == 0:
        _jsonl({"type": "done", "session": session_id, "ranking": [], "wav_paths": {}})
        return 0

    t_silence = min(0.20, max(0.05, silence_seconds / max(1, total_devices)))
    t_speak = min(0.80, max(0.20, speak_seconds / max(1, total_devices)))

    # First pick params, then measure noise during silence window.
    noise_by_device: dict[int, float] = {}
    params_by_device: dict[int, list[AudioParams]] = {}

    silence_deadline = time.monotonic() + silence_seconds
    _jsonl(
        {
            "type": "phase",
            "session": session_id,
            "phase": "silence",
            "remaining": silence_seconds,
        }
    )

    for idx, device in enumerate(devices, start=1):
        remaining = max(0.0, silence_deadline - time.monotonic())
        _jsonl(
            {
                "type": "progress",
                "session": session_id,
                "phase": "silence",
                "current": idx,
                "total": total_devices,
                "device_index": device.index,
                "remaining": remaining,
            }
        )

        candidates = _pick_params_with_fallback(device)
        params_by_device[device.index] = candidates
        if not candidates:
            _jsonl(
                {
                    "type": "device",
                    "session": session_id,
                    "status": "fail",
                    "device_index": device.index,
                    "hostapi": device.hostapi,
                    "name": device.name,
                    "is_default": device.is_default,
                    "err": "no openable params",
                }
            )
            continue

        # Measure noise on the first candidate.
        try:
            chunks = _open_and_read(
                device_index=device.index, params=candidates[0], seconds=t_silence
            )
            noise_rms = float(sum(_chunk_rms(c) for c in chunks) / max(1, len(chunks)))
            noise_by_device[device.index] = noise_rms
        except Exception as exc:
            _jsonl(
                {
                    "type": "device",
                    "session": session_id,
                    "status": "fail",
                    "device_index": device.index,
                    "hostapi": device.hostapi,
                    "name": device.name,
                    "is_default": device.is_default,
                    "err": f"noise read failed: {exc}",
                }
            )

    # Speak phase: measure speech metrics.
    speak_deadline = time.monotonic() + speak_seconds
    _jsonl(
        {
            "type": "phase",
            "session": session_id,
            "phase": "speak",
            "remaining": speak_seconds,
        }
    )

    results: dict[int, ScanResult] = {}
    speak_frames_by_device: dict[int, list[bytes]] = {}

    for idx, device in enumerate(devices, start=1):
        remaining = max(0.0, speak_deadline - time.monotonic())
        _jsonl(
            {
                "type": "progress",
                "session": session_id,
                "phase": "speak",
                "current": idx,
                "total": total_devices,
                "device_index": device.index,
                "remaining": remaining,
            }
        )

        candidates = params_by_device.get(device.index) or []
        if not candidates or device.index not in noise_by_device:
            if device.index not in results:
                penalty_monitor, penalty_dup = _compute_penalties(device)
                results[device.index] = ScanResult(
                    device=device,
                    status="fail",
                    params=None,
                    metrics=None,
                    score=None,
                    penalty_monitor=penalty_monitor,
                    penalty_dup=penalty_dup,
                    err="missing noise baseline",
                )
            continue

        noise_rms = noise_by_device[device.index]

        def measure_with_params(
            params: AudioParams, seconds: float
        ) -> tuple[ScanMetrics, list[bytes]]:
            chunks = _open_and_read(
                device_index=device.index, params=params, seconds=seconds
            )
            speech_rms, activity, zero_ratio, clip_ratio = _analyze_chunks(
                chunks, noise_rms=noise_rms
            )
            snr_like = (speech_rms - noise_rms) / (noise_rms + 1e-6)
            return (
                ScanMetrics(
                    noise_rms=noise_rms,
                    speech_rms=speech_rms,
                    snr_like=float(snr_like),
                    activity=float(activity),
                    zero_ratio=float(zero_ratio),
                    clip_ratio=float(clip_ratio),
                ),
                chunks,
            )

        best_params = candidates[0]
        best_metrics, best_chunks = measure_with_params(best_params, t_speak)

        # If the first candidate looks like silence, try fallback quickly.
        looks_silent = best_metrics.zero_ratio > 0.95 or best_metrics.speech_rms <= max(
            noise_rms * 1.2, 60.0
        )
        if looks_silent and len(candidates) >= 2:
            # Spend at most 40% of the per-device speak budget on fallback.
            fallback_seconds = max(0.12, min(t_speak * 0.4, t_speak))
            try:
                fallback_metrics, fallback_chunks = measure_with_params(
                    candidates[1], fallback_seconds
                )
                if fallback_metrics.snr_like > best_metrics.snr_like:
                    best_params = candidates[1]
                    best_metrics = fallback_metrics
                    best_chunks = fallback_chunks
            except Exception:
                pass

        penalty_monitor, penalty_dup = _compute_penalties(device)
        score = _score_device(metrics=best_metrics, device=device)

        results[device.index] = ScanResult(
            device=device,
            status="ok",
            params=best_params,
            metrics=best_metrics,
            score=score,
            penalty_monitor=penalty_monitor,
            penalty_dup=penalty_dup,
            err=None,
        )
        speak_frames_by_device[device.index] = best_chunks

        _jsonl(
            {
                "type": "device",
                "session": session_id,
                "status": "ok",
                "device_index": device.index,
                "hostapi": device.hostapi,
                "name": device.name,
                "is_default": device.is_default,
                "rate": best_params.rate,
                "channels": best_params.channels,
                "chunk": best_params.chunk,
                "noise_rms": round(best_metrics.noise_rms, 2),
                "speech_rms": round(best_metrics.speech_rms, 2),
                "snr_like": round(best_metrics.snr_like, 3),
                "activity": round(best_metrics.activity, 3),
                "zero_ratio": round(best_metrics.zero_ratio, 4),
                "clip_ratio": round(best_metrics.clip_ratio, 4),
                "penalty": {"monitor": penalty_monitor, "dup": penalty_dup},
                "score": round(score, 4),
            }
        )

    ok_results = [
        r for r in results.values() if r.status == "ok" and r.score is not None
    ]
    ok_results.sort(key=lambda r: float(r.score or 0.0), reverse=True)
    ranking = [r.device.index for r in ok_results]

    wav_paths: dict[str, str] = {}
    for r in ok_results[: max(0, top_k)]:
        if r.params is None:
            continue
        frames = speak_frames_by_device.get(r.device.index)
        if not frames:
            continue
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            prefix=f"voice-input-scan-{r.device.index}-",
            suffix=".wav",
        ) as wav_file:
            wav_path = Path(wav_file.name)
        try:
            _write_wav(wav_path, params=r.params, frames=frames)
            wav_paths[str(r.device.index)] = str(wav_path)
        except Exception:
            try:
                wav_path.unlink()
            except OSError:
                pass

    _jsonl(
        {
            "type": "done",
            "session": session_id,
            "ranking": ranking,
            "wav_paths": wav_paths,
        }
    )
    return 0


def _format_percent(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.0f}%"


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


class DeviceSelectorApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #main {
        width: 98%;
        height: 92%;
    }
    #device_table {
        height: 1fr;
        margin-top: 1;
        margin-bottom: 1;
    }
    #status {
        height: 4;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("a", "auto_scan", "Auto Scan"),
        ("d", "toggle_duration", "10/15s"),
        ("p", "replay_candidate", "Replay"),
        ("y", "accept_candidate", "Accept"),
        ("n", "next_candidate", "Next"),
        ("s", "save_candidate", "Save"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.devices: list[InputDevice] = list_input_devices_extended()
        self._scan_proc: subprocess.Popen[str] | None = None
        self._scan_session: int = 0
        self._scan_running: bool = False
        self._speak_seconds: int = 10

        self._results: dict[int, dict[str, Any]] = {}
        self._ranking: list[int] = []
        self._wav_paths: dict[int, str] = {}

        self._play_index: int = 0
        self._accepted_device: int | None = None
        self._save_enabled: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="main"):
            yield Static("", id="title")
            yield DataTable(id="device_table")
            yield Static("Press 'a' to auto scan all devices.", id="status")
        yield Footer()

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter" and self._save_enabled:
            self.action_save_candidate()
            event.stop()

    def on_mount(self) -> None:
        self._update_title()

        table = self.query_one(DataTable)
        table.add_columns(
            ("Index", "index"),
            ("HostAPI", "hostapi"),
            ("Name", "name"),
            ("Default", "default"),
            ("Status", "status"),
            ("Score", "score"),
            ("Params", "params"),
            ("NoiseRMS", "noise_rms"),
            ("SpeechRMS", "speech_rms"),
            ("SNR", "snr"),
            ("Act%", "act"),
            ("Zero%", "zero"),
            ("Clip%", "clip"),
            ("Penalty", "penalty"),
        )

        for device in self.devices:
            table.add_row(
                str(device.index),
                device.hostapi,
                device.name,
                "Yes" if device.is_default else "",
                "pending",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                key=str(device.index),
            )

        table.cursor_type = "row"
        if self.devices:
            default_row = 0
            for row_index, device in enumerate(self.devices):
                if device.is_default:
                    default_row = row_index
                    break
            table.cursor_coordinate = Coordinate(0, default_row)

    def _update_title(self) -> None:
        title = self.query_one("#title", Static)
        title.update(
            f"Voice Input Auto Selector (a: scan, d: 10/15s, y/n: accept/next, s: save) | speak={self._speak_seconds}s"
        )

    def set_status(self, message: str) -> None:
        status = self.query_one("#status", Static)
        status.update(message)

    def _reset_scan_state(self) -> None:
        self._results.clear()
        self._ranking = []
        self._wav_paths = {}
        self._play_index = 0
        self._accepted_device = None
        self._save_enabled = False

        table = self.query_one(DataTable)
        for device in self.devices:
            row_key = str(device.index)
            table.update_cell(row_key, "status", "pending")
            for col in (
                "score",
                "params",
                "noise_rms",
                "speech_rms",
                "snr",
                "act",
                "zero",
                "clip",
                "penalty",
            ):
                table.update_cell(row_key, col, "")

    def _stop_scan_worker(self) -> None:
        if self._scan_proc is None:
            return
        try:
            self._scan_proc.terminate()
        except Exception:
            pass
        self._scan_proc = None
        self._scan_running = False

    def action_toggle_duration(self) -> None:
        # Toggle speak duration.
        self._speak_seconds = 15 if self._speak_seconds == 10 else 10
        self._update_title()

        if self._scan_running:
            # Stop and restart from scratch.
            self.set_status("Restarting scan with new duration...")
            self._stop_scan_worker()
            self._reset_scan_state()
            self.action_auto_scan()
        else:
            self.set_status(
                f"Speak duration set to {self._speak_seconds}s. Press 'a' to scan."
            )

    def action_auto_scan(self) -> None:
        if self._scan_running:
            self.set_status("Scan already running. Press 'd' to restart with 10/15s.")
            return

        self._scan_session += 1
        session_id = self._scan_session
        self._reset_scan_state()
        self._scan_running = True

        self.set_status(
            f"Auto scan starting. Keep silent then speak for {self._speak_seconds}s... (press d to switch 10/15s)"
        )

        self.run_worker(
            lambda: self._run_scan_worker(session_id), thread=True, exclusive=True
        )

    def _run_scan_worker(self, session_id: int) -> None:
        cmd = [
            sys.executable,
            str(Path(__file__)),
            "--scan-all-worker",
            "--session-id",
            str(session_id),
            "--silence-seconds",
            "2",
            "--speak-seconds",
            str(self._speak_seconds),
            "--top-k",
            "3",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.call_from_thread(
                self.set_status, f"Failed to start scan worker: {exc}"
            )
            self._scan_running = False
            return

        self._scan_proc = proc

        def still_current() -> bool:
            return (
                self._scan_running
                and self._scan_session == session_id
                and self._scan_proc is proc
            )

        if proc.stdout is None:
            self.call_from_thread(self.set_status, "Scan worker has no stdout")
            self._scan_running = False
            return

        for line in proc.stdout:
            if not still_current():
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if msg.get("session") != session_id:
                continue
            self.call_from_thread(self._handle_worker_message, msg)

        if still_current():
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            rc = proc.returncode
            if rc not in (0, None):
                self.call_from_thread(self.set_status, f"Scan worker exited rc={rc}.")

        if still_current():
            self._scan_running = False
            self._scan_proc = None

    def _handle_worker_message(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "phase":
            phase = msg.get("phase")
            remaining = msg.get("remaining")
            if phase == "silence":
                self.set_status(
                    f"[Silence] Keep silent... remaining={_format_float(remaining, 1)}s"
                )
            elif phase == "speak":
                self.set_status(
                    f"[Speak] Speak continuously... remaining={_format_float(remaining, 1)}s"
                )
            return

        if kind == "progress":
            phase = msg.get("phase")
            current = msg.get("current")
            total = msg.get("total")
            remaining = msg.get("remaining")
            device_index = msg.get("device_index")
            self.set_status(
                f"[{phase}] {current}/{total} device={device_index} remaining={_format_float(remaining, 1)}s (d toggles 10/15s)"
            )
            return

        if kind == "device":
            status = msg.get("status")
            device_index_raw = msg.get("device_index")
            if device_index_raw is None:
                return
            device_index = int(device_index_raw)

            table = self.query_one(DataTable)
            row_key = str(device_index)

            if status == "fail":
                table.update_cell(row_key, "status", "fail")
                table.update_cell(row_key, "penalty", "")
                err = msg.get("err")
                self._results[device_index] = {"status": "fail", "err": err}
                return

            # ok
            table.update_cell(row_key, "status", "ok")

            score_value = msg.get("score")
            score = float(score_value) if score_value is not None else 0.0

            rate_raw = msg.get("rate")
            channels_raw = msg.get("channels")
            chunk_raw = msg.get("chunk")
            if rate_raw is None or channels_raw is None or chunk_raw is None:
                table.update_cell(row_key, "status", "fail")
                self._results[device_index] = {
                    "status": "fail",
                    "err": "missing rate/channels/chunk",
                }
                return

            rate = int(rate_raw)
            channels = int(channels_raw)
            chunk = int(chunk_raw)

            penalty = msg.get("penalty") or {}
            penalty_text = (
                f"mon={penalty.get('monitor', 0)} dup={penalty.get('dup', 0)}"
            )

            noise_value = msg.get("noise_rms")
            speech_value = msg.get("speech_rms")
            snr_value = msg.get("snr_like")
            activity_value = msg.get("activity")
            zero_value = msg.get("zero_ratio")
            clip_value = msg.get("clip_ratio")

            noise_rms = float(noise_value) if noise_value is not None else 0.0
            speech_rms = float(speech_value) if speech_value is not None else 0.0
            snr_like = float(snr_value) if snr_value is not None else 0.0
            activity = float(activity_value) if activity_value is not None else 0.0
            zero_ratio = float(zero_value) if zero_value is not None else 0.0
            clip_ratio = float(clip_value) if clip_value is not None else 0.0

            table.update_cell(
                row_key, "score", _format_float(score, 3), update_width=True
            )
            table.update_cell(row_key, "params", f"{rate}/{channels}ch/{chunk}")
            table.update_cell(row_key, "noise_rms", _format_float(noise_rms, 1))
            table.update_cell(row_key, "speech_rms", _format_float(speech_rms, 1))
            table.update_cell(row_key, "snr", _format_float(snr_like, 2))
            table.update_cell(row_key, "act", _format_percent(activity))
            table.update_cell(row_key, "zero", _format_percent(zero_ratio))
            table.update_cell(row_key, "clip", _format_percent(clip_ratio))
            table.update_cell(row_key, "penalty", penalty_text)

            self._results[device_index] = {
                "status": "ok",
                "score": score,
                "rate": rate,
                "channels": channels,
                "chunk": chunk,
                "noise_rms": noise_rms,
                "speech_rms": speech_rms,
            }
            return

        if kind == "done":
            ranking = [int(x) for x in (msg.get("ranking") or [])]
            wav_paths_raw: dict[str, str] = msg.get("wav_paths") or {}
            wav_paths = {int(k): v for k, v in wav_paths_raw.items()}

            self._ranking = ranking
            self._wav_paths = wav_paths
            self._scan_running = False
            self._scan_proc = None

            if not ranking:
                self.set_status("Scan finished: no usable devices found.")
                return

            # Highlight top candidate.
            table = self.query_one(DataTable)
            top = ranking[0]
            try:
                row_index = table.get_row_index(str(top))
                table.cursor_coordinate = Coordinate(0, row_index)
            except Exception:
                pass

            self._play_index = 0
            self.set_status(f"Scan finished. Top candidate={top}. Replaying once...")
            self.run_worker(
                lambda: self._play_current_candidate_once(), thread=True, exclusive=True
            )
            return

    def _current_candidate_device(self) -> int | None:
        if not self._ranking:
            return None
        if not (0 <= self._play_index < len(self._ranking)):
            return None
        return self._ranking[self._play_index]

    def _play_current_candidate_once(self) -> None:
        device_index = self._current_candidate_device()
        if device_index is None:
            self.call_from_thread(self.set_status, "No candidate to play")
            return

        wav_path = self._wav_paths.get(device_index)
        if not wav_path:
            self.call_from_thread(self.set_status, f"No wav for device {device_index}")
            return

        # Playback via subprocess to avoid breaking Textual terminal state.
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__)),
                "--play-from",
                wav_path,
                "--record-chunk",
                "1024",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            # Fallback to pw-play / paplay.
            for cmd in (["pw-play", wav_path], ["paplay", wav_path]):
                try:
                    fallback = subprocess.run(
                        cmd, capture_output=True, text=True, check=False
                    )
                    if fallback.returncode == 0:
                        self.call_from_thread(
                            self.set_status,
                            f"Played with {' '.join(cmd[:-1])}. Press y to accept (enables s), n to next.",
                        )
                        return
                except FileNotFoundError:
                    continue

            self.call_from_thread(
                self.set_status,
                f"Playback failed. rc={result.returncode}. Press n for next candidate.",
            )
            return

        self.call_from_thread(
            self.set_status,
            "Playback done. Press y to accept (enables s), n for next, p to replay.",
        )

    def action_replay_candidate(self) -> None:
        if not self._ranking:
            self.set_status("No candidates yet. Press 'a' to scan.")
            return
        self.run_worker(
            lambda: self._play_current_candidate_once(), thread=True, exclusive=True
        )

    def action_next_candidate(self) -> None:
        if not self._ranking:
            self.set_status("No candidates yet. Press 'a' to scan.")
            return
        if self._play_index + 1 >= min(3, len(self._ranking)):
            self.set_status("No more candidates to try (top3).")
            return
        self._play_index += 1
        self.set_status(f"Trying next candidate: {self._current_candidate_device()}...")
        self.run_worker(
            lambda: self._play_current_candidate_once(), thread=True, exclusive=True
        )

    def action_accept_candidate(self) -> None:
        device_index = self._current_candidate_device()
        if device_index is None:
            self.set_status("No candidate to accept.")
            return
        self._accepted_device = device_index
        self._save_enabled = True

        r = self._results.get(device_index) or {}
        params_text = ""
        if r.get("status") == "ok":
            params_text = f" {r.get('rate')}/{r.get('channels')}ch/{r.get('chunk')}"

        self.set_status(
            f"Accepted device={device_index}.{params_text} Press 's' to save, 'n' to try next."
        )

    def action_save_candidate(self) -> None:
        if not self._save_enabled or self._accepted_device is None:
            self.set_status("Press 'y' to accept a candidate first.")
            return

        accepted = self._accepted_device
        r = self._results.get(accepted)
        if not r or r.get("status") != "ok":
            self.set_status("Accepted device has no valid scan result.")
            return

        updates = {
            "DEVICE_INDEX": str(accepted),
            "RECORD_RATE": str(int(r["rate"])),
            "RECORD_CHANNELS": str(int(r["channels"])),
            "RECORD_CHUNK": str(int(r["chunk"])),
        }
        write_env_file(ENV_PATH, updates)

        self.set_status(
            f"Saved DEVICE_INDEX={accepted} ({r['rate']}Hz, {r['channels']}ch, chunk={r['chunk']})."
        )


def record_to_wav(
    output_path: Path,
    *,
    device_index: int,
    record_seconds: float,
    params: AudioParams,
) -> float:
    """Record audio to WAV; return average RMS."""

    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=params.channels,
            rate=params.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=params.chunk,
        )

        frames: list[bytes] = []
        rms_values: list[float] = []

        num_chunks = max(1, int(record_seconds * params.rate / params.chunk))
        for _ in range(num_chunks):
            data = stream.read(params.chunk, exception_on_overflow=False)
            frames.append(data)
            rms_values.append(_chunk_rms(data))

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(params.channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(params.rate)
            wf.writeframes(b"".join(frames))

        return float(sum(rms_values) / max(1, len(rms_values)))
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()


def play_wav(input_path: Path, *, chunk: int) -> None:
    p = pyaudio.PyAudio()
    try:
        with wave.open(str(input_path), "rb") as wf:
            output = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                frames_per_buffer=chunk,
            )
            try:
                while True:
                    data = wf.readframes(chunk)
                    if not data:
                        break
                    output.write(data)
            finally:
                output.stop_stream()
                output.close()
    finally:
        p.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("--scan-all-worker", action="store_true")
    parser.add_argument("--scan-one-worker", action="store_true")
    parser.add_argument("--session-id", type=int, default=0)
    parser.add_argument("--silence-seconds", type=float, default=2.0)
    parser.add_argument("--speak-seconds", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sample-seconds", type=float, default=0.4)
    parser.add_argument("--device-name", type=str, default="")
    parser.add_argument("--hostapi", type=str, default="")
    parser.add_argument("--max-input-channels", type=int, default=1)
    parser.add_argument("--default-sample-rate", type=int, default=44100)
    parser.add_argument("--dup-rank", type=int, default=0)
    parser.add_argument("--is-default", action="store_true")
    parser.add_argument("--monitor-like", action="store_true")

    parser.add_argument("--record-to", type=Path, default=None)
    parser.add_argument("--play-from", type=Path, default=None)
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--record-seconds", type=float, default=3.0)
    parser.add_argument("--record-rate", type=int, default=None)
    parser.add_argument("--record-channels", type=int, default=None)
    parser.add_argument("--record-chunk", type=int, default=None)

    args = parser.parse_args()

    if args.scan_one_worker:
        if args.device_index is None:
            return 2
        return run_scan_one_worker(
            session_id=int(args.session_id),
            device_index=int(args.device_index),
            hostapi=str(args.hostapi),
            name=str(args.device_name),
            max_input_channels=int(args.max_input_channels),
            default_sample_rate=int(args.default_sample_rate),
            is_default=bool(args.is_default),
            dup_rank=int(args.dup_rank),
            monitor_like=bool(args.monitor_like),
            sample_seconds=float(args.sample_seconds),
            allow_fallback=True,
        )

    if args.scan_all_worker:
        return run_scan_all_worker(
            session_id=int(args.session_id),
            silence_seconds=float(args.silence_seconds),
            speak_seconds=float(args.speak_seconds),
            top_k=int(args.top_k),
        )

    if args.play_from is not None:
        play_wav(args.play_from, chunk=args.record_chunk or 1024)
        print("PLAY_OK")
        return 0

    if args.record_to is not None:
        if args.device_index is None:
            print("--device-index is required with --record-to")
            return 2

        # Minimal probe: pick first openable params.
        devices = [
            d for d in list_input_devices_extended() if d.index == args.device_index
        ]
        if not devices:
            print("unknown device")
            return 2

        candidates = _pick_params_with_fallback(devices[0])
        if not candidates:
            print("no openable params")
            return 2

        params = candidates[0]
        params = AudioParams(
            rate=args.record_rate or params.rate,
            channels=args.record_channels or params.channels,
            chunk=args.record_chunk or params.chunk,
        )

        rms = record_to_wav(
            args.record_to,
            device_index=args.device_index,
            record_seconds=args.record_seconds,
            params=params,
        )
        print(
            f"REC_OK rate={params.rate} channels={params.channels} chunk={params.chunk} rms={rms:.0f}"
        )
        return 0

    DeviceSelectorApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
