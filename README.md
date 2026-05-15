# 语言模型高效推理实验：Baseline vs StreamingLLM
Author：毛颖琦  524531910018

## 1. 实验内容
- 模型：EleutherAI/pythia-70m
- 数据集：wikitext-2-raw-v1 (validation split)
- 方法：baseline（无压缩） / streamingllm（sink+window动态KV压缩）
- 输入长度：2048 token
- 生成长度：256 token
- 指标：PPL、TTFT、TPOT、Throughput、峰值显存、KV Cache、Attention FLOPs

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

## 5. 指标解释
- PPL：困惑度，数值越小表示生成质量越高
- TTFT：生成第一个token的总耗时（秒）
- TPOT：每个输出token的平均耗时（毫秒）
- Throughput：每秒生成的token数量
- Peak Memory：GPU峰值显存（CPU环境下为0）
- KV Cache Memory：压缩后KV缓存实际占用内存（MB）
- Attention FLOPs：注意力模块计算量估算（G）

## 6. 实验结果
| Dataset | Method | Budget | PPL | TTFT (s) | TPOT (ms/token) | Throughput (tok/s) | Peak Memory (MB) | KV Cache (MB) | Attention FLOPs (G) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wikitext | baseline | 100% | 43.2805 | 1.0502 | 15.58 | 50.96 | 0.0 | 48.00 | 28.991 |
| wikitext | streamingllm | 20% | 54.4141 | 1.4400 | 10.22 | 64.75 | 0.0 | 3.09 | 26.414 |
| wikitext | streamingllm | 30% | 54.4141 | 1.7024 | 11.32 | 55.79 | 0.0 | 3.09 | 26.736 |
| wikitext | streamingllm | 50% | 54.4141 | 1.7579 | 11.38 | 54.95 | 0.0 | 3.09 | 27.380 |

## 7. 结果分析
- Baseline 保留完整KV缓存（48MB），PPL最优，为 43.2805
- StreamingLLM 可将KV缓存压缩至 3.09MB，内存减少 93.5%
- TPOT 从 15.58ms 显著降低至 9~11ms，推理速度大幅提升
- Throughput 从 50.96 tok/s 提升至 54~65 tok/s
- 预算比例（0.2/0.3/0.5）对缓存大小无影响，均受限于 sink+window 结构
- CPU环境下显存显示为0，内存压缩效果不受影响

## 8. 注意事项
- 首次运行需联网下载模型与数据集
- CPU环境运行速度较慢，可降低 input_length 与 generate_length
- 实验结果自动保存至 results_streamingllm 文件夹
- budget_ratio 过低会导致PPL上升，生成质量下降
- 本实验所有方法均无训练，推理时动态压缩KV缓存
