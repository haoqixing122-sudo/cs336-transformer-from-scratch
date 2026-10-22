# CS336 Transformer From Scratch

参照 CS336 Assignment 1 学习和实现的小型自回归 Transformer 项目，
包含 BPE 分词、语料预处理、模型训练、checkpoint 恢复和文本生成。

## 实现内容

- 字节级 BPE 训练与 Tokenizer 编码、解码
- Embedding、Linear、RMSNorm、RoPE、SwiGLU
- 因果多头自注意力与 Transformer 语言模型
- 交叉熵、AdamW、梯度裁剪和学习率调度
- checkpoint 保存与恢复
- 贪心生成、温度采样和 Top-P 采样

## 当前实验配置

| 参数 | 值 |
|---|---:|
| 词表大小 | 4096 |
| 上下文长度 | 256 |
| 隐藏维度 | 256 |
| Transformer 层数 | 4 |
| 注意力头数 | 4 |
| 前馈中间维度 | 704 |
| 参数量 | 5,310,720 |
| Batch size | 4 |

## 数据与结果

数据来自 TinyStories：
https://huggingface.co/datasets/roneneldan/TinyStories

- BPE 使用初始小样本的训练部分学习，未使用验证文本。
- 扩大实验使用约 50 MiB 训练文本和约 5 MiB 官方验证文本。
- 训练集约 1319 万 token，验证集约 131 万 token。
- 累计训练 30000 步后，固定抽取的 10 个验证 batch 平均 loss 约为 2.055。
- 验证指标未覆盖完整验证集，不作为正式基准成绩。

模型能够生成基本故事，但仍存在角色混淆、情节跳跃和重复表达。

## 运行说明

从项目根目录使用模块方式启动：

    python -m cs336_basics.train_bpe
    python -m cs336_basics.main_train --help
    python -m cs336_basics.inference --help

训练前需要准备语料、训练分词器并生成 token 二进制文件。
不同实验需显式指定对应路径和模型参数。
完整训练数据、模型权重和本地日志不包含在仓库中。

## 参考与辅助说明

参考仓库：
https://github.com/chaser026/cs336-1

本项目用于个人学习，部分实现参考上述仓库，
代码排查、部分修改及实验配置使用了 AI 辅助。
目前未完成课程全部官方测试的核验。