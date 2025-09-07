# Qwen2.5 GSM8K 推理能力评估

这个项目用于评估 Qwen2.5 模型在 GSM8K 数据集上的数学推理能力。

## 项目结构

- `src/一训练前评价模型.py`: 训练前模型评估脚本
- `src/二微调模型.py`: 模型微调脚本
- `src/三训练后评价模型.py`: 训练后模型评估脚本
- `ds_config.json`: DeepSpeed配置文件
- `requirements.txt`: 项目依赖

## 依赖安装

```bash
pip install -r requirements.txt
```

## 使用流程

1. **训练前评估**
   ```bash
   python src/一训练前评价模型.py
   ```

2. **模型微调**
   ```bash
   python src/二微调模型.py
   ```

3. **训练后评估**
   ```bash
   python src/三训练后评价模型.py
   ```

## 输出结果

评估结果将保存在以下文件中：

1. `一评价结果训练前.csv`: 包含每个样本的详细结果
   - 问题
   - 真实答案
   - 预测答案
   - 模型完整输出
   - 是否正确
   - ROUGE-L分数

## 实现细节

1. 使用 Hugging Face Transformers 库加载 Qwen2.5 模型
2. 使用 GSM8K 数据集的测试集进行评估
3. 采用 "Let's think step by step" 提示词引导模型逐步推理
4. 从模型输出中提取最终答案并与标准答案比较
5. 计算准确率和ROUGE-L分数并保存结果

## 评估结果

- 训练前ROUGE-L分数: 0.17
- 训练后ROUGE-L分数: 0.29
- 提升幅度: 0.12 (70.6% 相对提升)

这表明通过微调，模型在GSM8K数据集上的推理能力得到了显著提升。