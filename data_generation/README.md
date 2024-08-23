# 生成用于自蒸馏的聊天数据
我们使用 vLLM 来实现批量生成。首先，安装依赖项：
```bash
pip install vllm openai
```

## 启动服务器

```bash
python -m vllm.entrypoints.openai.api_server \
    --model YOUR_MODEL_NAME --port 8000
```
你也可以启动多个服务器并使用不同的端口来实现并行生成。在 `generate.py` 中，我们会扫描从 8000 到 8009 的端口以查找可用的服务器。你可以修改代码以使用其他端口。

## 生成数据
以下命令将让模型从 `DATA_PATH` 中每个样本的第一个提示开始生成，这适用于能够在对话中扮演两个角色的模型（例如 `Zephyr 7B`）。如果你想使用每个样本中的所有提示与模型反复对话，请使用 `--chat` 参数。`--chat` 模式适用于更多的模型，但由于重复计算，生成时间可能会更长（欢迎贡献更好的实现）。

```bash
python generate.py --data_path YOUR_DATA_PATH --output_path YOUR_OUTPUT_PATH --num_threads NUM_THREADS --max_tokens YOUR_MAX_TOKENS --temperature YOUR_TEMPERATURE
```

## （可选）格式化数据
使用 `--chat` 生成的数据文件将遵循 `ShareGPT` 格式。你可以使用以下命令将不使用 `--chat` 生成的文本转换为相同的格式：
```bash
python convert_to_sharegpt.py --input_path YOUR_INPUT_PATH --model_name YOUR_MODEL_NAME --output_path YOUR_OUTPUT_PATH
```