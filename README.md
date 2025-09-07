# Qwen2.5 GSM8K 推理能力评估

这个项目用于评估 Qwen2.5 模型在 GSM8K 数据集上的数学推理能力。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 运行评估

```bash
python evaluate_qwen.py
```

## 输出结果

评估结果将保存在以下文件中：

1. `evaluation_results.json`：包含汇总统计信息
   - 模型名称
   - 总问题数
   - 正确预测数
   - 准确率
   - 平均ROUGE-L分数

2. `detailed_results.csv`：包含每个样本的详细结果
   - 问题
   - 真实答案
   - 预测答案
   - 模型完整输出
   - 是否正确
   - ROUGE-L分数

## 实现细节

1. 使用 Hugging Face Transformers 库加载 Qwen2.5 模型
2. 使用 GSM8K 数据集的测试集进行评估
3. 随机选择50个样本进行评估（如果数据集超过50个样本）
4. 采用 "Let's think step by step" 提示词引导模型逐步推理
5. 从模型输出中提取最终答案并与标准答案比较
6. 计算准确率和ROUGE-L分数并保存结果