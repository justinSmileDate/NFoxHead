# LLM Judge

## 安装

| [Guide](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md)

## 使用

我们报告了 NFoxHead X Vicuna v1.3 7/13/33b 在单个 A100 上运行 3 次的结果，数据保存在 `./data/mt_bench/model_answer/` 中。原始设置为：`temperature`（已弃用，使用默认的 LLM Judge 设置）、`posterior_threshold=0.09`、`posterior_alpha=0.3`。

- 运行基准测试



```
export CUDA_VISIBLE_DEVICES=0 # set the GPU id
python gen_model_answer_NFoxHead.py  --model-path lmsys/vicuna-7b-v1.3 --model-id lmsys/vicuna-7b-v1.3
python gen_model_answer_NFoxHead.py  --model-path lmsys/vicuna-13b-v1.3 --model-id lmsys/vicuna-13b-v1.3
python gen_model_answer_NFoxHead.py  --model-path lmsys/vicuna-33b-v1.3 --model-id lmsys/vicuna-33b-v1.3
```

- 运行基线测试：将 `gen_model_answer_NFoxHead.py` 替换为 `gen_model_answer_baseline.py`（请注意，我们仅实现了贪婪推理用于时间比较。

- 查询结果

```
export OPENAI_API_KEY=$OPENAI_API_KEYs # set the OpenAI API key
python gen_judgement.py --model-list NFoxHead-vicuna-7b-v1.3-0-temperature-0.0-posterior_threshold-0.09-posterior_alpha-0.3 
```

- 显示结果

要获取 GPT-4 对 Vicuna-7b 的评估结果（包括 Huggingface 贪婪推理 | Huggingface 采样 | Medusa 采样），请运行以下命令：


```
python show_result.py
```
