# OmniParser PKU Winter Camp Enhancement

本项目是针对 [OmniParser](https://github.com/microsoft/OmniParser) 的功能增强版，结合了主流多模态大模型（如 GPT-4o, Gemini 等），实现了精准的 UI 指导执行（Instruction Following）。


## 🛠️ 安装指南

1. **环境准备**：
   建议使用 Python 3.10+ 环境。安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. **模型权重**：
   请确保 `weights/` 目录下包含 OmniParser 所需的 YOLO 和 Florence-2 权重文件：
   - `weights/icon_detect/model.pt`
   - `weights/icon_caption_florence`

3. **配置 API**：
   在根目录下创建 `.env` 文件，填入你的多模态模型 API 信息：
   ```env
   OPENAI_API_KEY=your_sk_key
   OPENAI_BASE_URL=https://api.your-provider.com/v1
   OPENAI_MODEL_NAME=gpt-4o
   ```

## 🚀 启动运行

直接启动 Gradio 服务：

```bash
python gradio_instruction.py
```

启动后访问 `http://localhost:7862` 即可进入交互界面。

## 📊 使用说明

1. **上传截图**：上传目标网页或应用的截图。
2. **输入指令**：例如 “点击左上角的登录按钮” 
3. **参数微调**：
   - 对于图标密集的界面（如 Word），调低 `IOU Threshold` (如 0.15) 以减少漏检。
   - 对于识别过少的情况，调低 `Box Threshold`。
4. **获取结果**：红色方框标记即为模型判定的点击目标，下方会显示详细的推理过程及 BBox 坐标。

---

