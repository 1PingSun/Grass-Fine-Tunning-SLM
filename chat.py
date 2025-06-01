from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
    user_input = input("🧑 你：")
    if user_input.lower() == "exit":
        print("👋 再見！")
        break

    chat_history.append(f"User: {user_input}")
    prompt = "\n".join(chat_history) + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip().split("User:")[0].strip()
    print("🤖 模型：", response)
    chat_history.append(f"Assistant: {response}")
