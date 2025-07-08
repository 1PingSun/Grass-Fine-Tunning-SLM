from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import pygame
import pyaudio       
from start_recording import sr

def ttsc(text):
    config = XttsConfig()
    config.load_json("XTTS-v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="XTTS-v2/", eval=True)
    # 确保使用GPU加速（如果可用）
    if torch.cuda.is_available():
        model.cuda()

    try:
        outputs = model.synthesize(
            text,
            config,
            speaker_wav="test.wav",
            gpt_cond_len=3,
            language="zh-cn",
        )
        # 将音频保存到文件
        import scipy
        scipy.io.wavfile.write("output_speech.wav", rate=24000, data=outputs["wav"])
        return "output_speech.wav"
        #print("语音合成成功，已保存到output_speech.wav")
    except AttributeError as e:
        if "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
            print("错误：transformers库版本过高，请降级到4.49.0版本")
            print("请运行: pip install transformers==4.49.0")
        else:
            raise e




model_path = "Qwen2.5-0.5B-Counseling"  

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # 若 GPU 支援 FP16，可使用。否則改為 torch.float32
    device_map="auto",
    trust_remote_code=True
)
chat_history = []

print("💬 模型已啟動，輸入 'exit' 結束聊天。")
while True:
    # user_input = input("🧑 你：")
    input("請按下 Enter 開始錄音並辨識...")
    user_input = sr()  # 按下 Enter 後才執行這行
    print("辨識結果：", user_input)
    if user_input.lower() == "exit":
        print("👋 再見！")
        break

    chat_history.append(f"User: {user_input}")
    prompt = "\n".join(chat_history) + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip().split("User:")[0].strip()
    wavload = ttsc(response)
    print(response)
    pygame.mixer.init()
    pygame.mixer.music.load(wavload)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    #print("🤖 模型：", response)
    chat_history.append(f"Assistant: {response}")
