# 语言模型效率实验：Baseline与StreamingLLM对比研究
Author：毛颖琦  524531910018

## 1. 实验内容
- 模型选择：EleutherAI/pythia-70m
- 实验数据集：wikitext-2-raw-v1 (validation split)
- 实验方法：baseline（无压缩） / streamingllm（sink+window动态KV压缩）
- 输入长度：2048 token
- 生成长度：256 token
- 评价的指标：涵盖生成质量（PPL）、推理速度（TTFT、TPOT、Throughput）及资源占用（峰值显存、KV Cache内存、Attention FLOPs）六大核心指标

## 2. 安装依赖
```bash
pip install torch transformers datasets accelerate sentencepiece protobuf pandas
```

## 3. 运行方式
标准实验命令：
```bash
python run_experiment.py --input_length 2048 --generate_length 256 --ppl_eval_tokens 256 --runs 2 --budget_ratios 0.2 0.3 0.5 --sink_size 4 --window_size 128
```

快速测试命令：
```bash
python run_experiment.py --datasets wikitext --input_length 2048 --generate_length 64 --ppl_eval_tokens 64 --runs 1 --budget_ratios 0.3
```

## 4. StreamingLLM 实现说明
1. 固定保留开头 sink_size=4 个 sink token，保证全局语义依赖
2. 固定保留最近 window_size=128 个 token，保证生成连贯性
3. 中间历史token直接丢弃，无需计算注意力权重，无训练、无额外开销
4. 所有层采用相同压缩策略，解码阶段每步动态裁剪KV cache
5. 手动传入 position_ids 保证位置编码不错乱
6. budget_ratio 控制总保留比例，最小值不低于 sink+window 长度



## 5. 实验结果
| Dataset | Method | Budget | PPL | TTFT (s) | TPOT (ms/token) | Throughput (tok/s) | Peak Memory (MB) | KV Cache (MB) | Attention FLOPs (G) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wikitext | baseline | 100% | 43.2805 | 1.0502 | 15.58 | 50.96 | 0.0 | 48.00 | 28.991 |
| wikitext | streamingllm | 20% | 54.4141 | 1.4400 | 10.22 | 64.75 | 0.0 | 3.09 | 26.414 |
| wikitext | streamingllm | 30% | 54.4141 | 1.7024 | 11.32 | 55.79 | 0.0 | 3.09 | 26.736 |
| wikitext | streamingllm | 50% | 54.4141 | 1.7579 | 11.38 | 54.95 | 0.0 | 3.09 | 27.380 |

## 6. 结果分析
- Baseline 保留完整KV缓存（48MB），PPL最优，为 43.2805
- StreamingLLM 可将KV缓存压缩至 3.09MB，内存减少 93.5%
- TPOT 从 15.58ms 显著降低至 9~11ms，推理速度大幅提升
- Throughput 从 50.96 tok/s 提升至 54~65 tok/s
- 预算比例（0.2/0.3/0.5）对缓存大小无影响，均受限于 sink+window 结构
- CPU环境下显存显示为0，内存压缩效果不受影响

## 7. 需要注意的部分
- CPU环境运行速度较慢，可降低 input_length 与 generate_length
- 实验结果自动保存至 results_streamingllm 文件夹
- budget_ratio 过低会导致PPL上升，生成质量下降
- 本实验所有方法均无训练，推理时动态压缩KV缓存
