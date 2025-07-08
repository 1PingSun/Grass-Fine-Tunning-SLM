import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import threading

def sr():
    SAMPLE_RATE = 16000
    CHANNELS = 1

    print("🎤 按 Enter 開始錄音")
    input()

    print("錄音中，按 Enter 停止錄音")

    # 開一個執行緒讓使用者按 Enter 停止錄音
    stop_flag = threading.Event()
    def wait_enter():
        input()
        stop_flag.set()

    threading.Thread(target=wait_enter).start()

    frames = []

    while not stop_flag.is_set():
        data = sd.rec(int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
        sd.wait()
        frames.append(data)

    audio = b''.join([f.tobytes() for f in frames])
    import numpy as np
    audio_np = np.frombuffer(audio, dtype='int16')

    wav.write("recorded.wav", SAMPLE_RATE, audio_np)

    print("✅ 錄音結束，開始語音辨識...")

    model = whisper.load_model("base")
    result = model.transcribe("recorded.wav", language='zh')

    print("🗣️ 辨識結果：")
    # print(result["text"])
    return result["text"]

