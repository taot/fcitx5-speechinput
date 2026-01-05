import pyaudio
import wave
import sys


def list_audio_devices():
    """列出所有可用的音频设备"""
    p = pyaudio.PyAudio()
    print("\n=== 可用的音频设备 ===")
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    input_devices = []
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            print(f"设备 {i}: {device_info.get('name')}")
            print(f"  - 最大输入声道: {device_info.get('maxInputChannels')}")
            print(f"  - 默认采样率: {device_info.get('defaultSampleRate')}")
            input_devices.append(i)

    p.terminate()
    return input_devices


def test_device_settings(device_index=None):
    """测试设备是否支持指定的参数"""
    p = pyaudio.PyAudio()

    test_configs = [
        {"rate": 44100, "channels": 1, "format": pyaudio.paInt16},
        {"rate": 48000, "channels": 1, "format": pyaudio.paInt16},
        {"rate": 16000, "channels": 1, "format": pyaudio.paInt16},
        {"rate": 44100, "channels": 2, "format": pyaudio.paInt16},
    ]

    print("\n=== 测试设备参数支持 ===")
    for config in test_configs:
        try:
            supported = p.is_format_supported(
                config["rate"],
                input_device=device_index,
                input_channels=config["channels"],
                input_format=config["format"]
            )
            print(f"✓ 支持: 采样率={config['rate']}, 声道={config['channels']}")
        except ValueError as e:
            print(f"✗ 不支持: 采样率={config['rate']}, 声道={config['channels']}")

    p.terminate()


def record_audio(device_index=None):
    """录音主函数"""
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    try:
        print(f"\n=== 开始录音 ===")
        print(f"使用设备: {device_index if device_index is not None else '默认设备'}")
        print(f"录音时长: {RECORD_SECONDS} 秒")
        print("请对着麦克风说话...")

        # 打开音频流
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )

        frames = []

        # 录音循环,显示进度
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

                # 显示进度
                progress = (i + 1) / (RATE / CHUNK * RECORD_SECONDS) * 100
                print(f"\r录音进度: {progress:.1f}%", end='')

            except Exception as e:
                print(f"\n读取音频数据时出错: {e}")
                break

        print("\n录音完成!")

        # 停止并关闭音频流
        stream.stop_stream()
        stream.close()

        # 保存为 WAV 文件
        wf = wave.open(OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(f"✓ 音频已保存到 {OUTPUT_FILENAME}")

    except Exception as e:
        print(f"\n✗ 录音失败: {e}")
        print("\n可能的原因:")
        print("1. 麦克风未连接或未启用")
        print("2. 没有录音权限(检查系统设置)")
        print("3. 设备正被其他程序占用")
        print("4. 不支持当前的音频参数")

    finally:
        p.terminate()


def main():
    print("=== 音频录制诊断工具 ===")

    # 1. 列出所有设备
    input_devices = list_audio_devices()

    if not input_devices:
        print("\n✗ 错误: 未检测到任何输入设备!")
        print("请检查:")
        print("- 麦克风是否已连接")
        print("- 系统是否识别到麦克风")
        print("- 驱动程序是否正常")
        return

    # 2. 测试默认设备
    test_device_settings()

    # 3. 选择设备
    print("\n=== 选择录音设备 ===")
    print("输入设备编号,或直接按回车使用默认设备:")

    try:
        user_input = input("> ").strip()
        device_index = int(user_input) if user_input else None
    except ValueError:
        device_index = None

    # 4. 开始录音
    record_audio(device_index)


if __name__ == "__main__":
    main()