from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 載入模型
model_path = "Qwen2.5-0.5B-Counseling"  

print("🔄 正在載入模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # 若 GPU 支援 FP16，可使用。否則改為 torch.float32
    device_map="auto",
    trust_remote_code=True
)

# 初始化聊天歷史
chat_history = []

print("💬 模型已啟動，輸入 'exit' 結束聊天。")
print("=" * 50)

while True:
    # 獲取用戶輸入
    user_input = input("🧑 你：")
    
    # 檢查是否要退出
    if user_input.lower() == "exit":
        print("👋 再見！")
        break
    
    # 如果輸入為空，跳過
    if not user_input.strip():
        continue
    
    # 將用戶輸入添加到聊天歷史
    chat_history.append(f"User: {user_input}")
    
    # 構建提示詞
    prompt = "\n".join(chat_history) + "\nAssistant:"
    
    # 將輸入轉換為模型可處理的格式
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回應
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,  # 增加生成長度以獲得更完整的回應
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解碼並提取回應
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip().split("User:")[0].strip()
    
    # 顯示模型回應
    print(f"🤖 模型：{response}")
    print("-" * 30)
    
    # 將模型回應添加到聊天歷史
    chat_history.append(f"Assistant: {response}")
    
    # 限制聊天歷史長度，避免過長影響性能
    if len(chat_history) > 20:  # 保留最近10輪對話
        chat_history = chat_history[-20:]
