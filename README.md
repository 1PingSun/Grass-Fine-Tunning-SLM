# Grass-Fine-Tunning-SLM

基於 Qwen2.5-0.5B 的語音對話系統，具備語音辨識、文字生成和語音合成功能。

## ✨ 功能特色

- 🎤 **語音識別**：將語音輸入轉換為文字
- 🤖 **智能對話**：使用 Qwen2.5-0.5B-Counseling 模型生成回應
- 🔊 **語音合成**：使用 XTTS-v2 將文字轉換為自然語音
- 🎵 **語音播放**：自動播放合成的語音回應

## 🛠️ 安裝步驟

### 1. 克隆專案
```bash
git https://github.com/1PingSun/Grass-Fine-Tunning-SLM.git
cd Grass-Fine-Tunning-SLM
```

### 2. 設置 venv
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows
```

### 3. 安裝 lib
```bash
pip install -r requirements.txt
```

### 4. 下載模型

#### 下載微調後的 Qwen2.5-0.5B-Counseling 輔導機器人模型
```bash
git clone https://huggingface.co/Rayifelse/Qwen2.5-0.5B-Counseling
```

#### 下載 XTTS-v2 語音合成模型
```bash
git clone https://huggingface.co/coqui/XTTS-v2
```

### 5. 準備語音文件
將你希望克隆的語音文件命名為 `test.wav` 並放置在專案根目錄中。

## 🚀 使用方法

### 1. 啟動語音對話系統
```bash
source .venv/bin/activate
python chat.py
```

### 2. 開始對話
- 程式啟動後，按 Enter 鍵開始錄音
- 說話完成後，系統會自動識別語音並生成回應
- AI 的回應會以語音形式播放
- 輸入 "exit" 結束對話

## 📁 專案結構

```
.
├── chat.py                    # 主要對話程式
├── start_recording.py         # 語音識別模組
├── requirements.txt           # Python 依賴包
├── test.wav                   # 參考語音文件（需自行提供）
├── Qwen2.5-0.5B-Counseling/   # 對話模型目錄
├── XTTS-v2/                   # 語音合成模型目錄
└── README.md                  # 專案說明文件
```

## 💻 核心程式碼說明

### `chat.py` - 主程式
- **語音識別**：使用 `start_recording.py` 中的 `sr()` 函數
- **文字生成**：使用 Qwen2.5-0.5B-Counseling 模型處理對話
- **語音合成**：使用 XTTS-v2 模型將回應轉換為語音
- **對話管理**：維護對話歷史，確保上下文連貫性

### 主要功能模組

#### 1. 語音合成函數 `ttsc(text)`
```python
def ttsc(text):
    # 載入 XTTS-v2 模型配置
    # 使用參考語音進行聲音克隆
    # 生成並保存語音文件
```

#### 2. 對話主循環
```python
while True:
    user_input = sr()  # 語音識別
    # 生成 AI 回應
    response = model.generate(...)
    # 語音合成並播放
    ttsc(response)
```

## 📦 依賴套件

主要依賴包括：
- `transformers==4.52.4` - Hugging Face 模型庫
- `TTS==0.22.0` - Coqui TTS 語音合成
- `torch==2.1.2` - PyTorch 深度學習框架
- `pygame` - 音頻播放
- `pyaudio` - 音頻錄製
- `openai-whisper` - 語音識別
- `scipy` - 科學計算庫
- `LLaMA-Factory` - 微調 LLM 

## ⚙️ 系統需求

- Python 3.8+
- 麥克風（用於語音輸入）
- 喇叭或耳機（用於語音輸出）
- 建議：NVIDIA GPU（加速模型推理）

## 故障排除

### 常見問題

1. **transformers 版本錯誤**
   ```bash
   pip install transformers==4.49.0
   ```

2. **音頻設備問題**
   - 確保麥克風權限已啟用
   - 檢查音頻輸出設備連接

3. **模型載入失敗**
   - 確認模型目錄路徑正確
   - 檢查磁碟空間是否充足

## 授權

本專案遵循 MIT 授權條款。

## 貢獻

歡迎提交 Issues 和 Pull Requests！

## 聯絡

如有問題，請透過 GitHub Issues 聯絡。 
