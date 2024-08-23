
我们使用 [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/0cd24d711fe90d0c1aae5bde03fe98ee48ae52f8/alpaca_eval.json)数据集，在 `heads_accuracy.py` 中评估每个头部在生成过程中的准确性。
```
python heads_accuracy.py --model_path 'FasterDecoding/medusa-vicuna-7b-v1.3' --model_name 'medusa-vicuna-7b-v1.3' --medusa_num_heads 5 --data_path '../../data/alpaca_eval.json'
```


要创建树并绘制树图（需要 `pygraphviz` 软件包），请运行：

```
python gen_results.py --accuracy-path '../../data/medusa-vicuna-7b-v1.3_heads_accuracy.pt' --output-path '../../data/graph.jpg'
```

如果要使用该树，请将生成的树（以嵌套元组的形式）添加到 `../model/NFoxHead_choices.py` 中。
