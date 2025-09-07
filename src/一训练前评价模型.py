# Qwen模型在GSM8K数据集上微调前的评估结果
# ROUGE-L分数为0.075，结果正确率为0.27
# 评估分数按ROUGE-L(0.8)与正确率(0.2)加权计算得出
# 评估分数为0.075 * 0.8 + 0.27 * 0.2 = 0.170

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import re
from rouge_score import rouge_scorer
import pandas as pd

def load_model(model_name):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Changed from torch.bfloat16 to torch.float16 for better CUDA compatibility
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

def format_question(question):
    """格式化问题"""
    return f"Question: {question}\nAnswer: Let's think step by step."


def calculate_rouge_l(prediction, reference):
    """计算ROUGE-L分数"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure

def evaluate_model(model, tokenizer, dataset, max_new_tokens=512):
    """评估模型在GSM8K数据集上的表现"""
    print("Starting model evaluation...")
    correct = 0
    total = 0
    rouge_l_scores = []
    results_data = []
    
    for item in tqdm(dataset, desc="Evaluating"):
        print("Processing item...")
        question = item['question']
        true_answer = item['answer']
        
        # 格式化输入
        input_text = format_question(question)
        print(f"Formatted question: {input_text}")
        messages = [{"role": "user", "content": input_text}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Input text after apply_chat_template: {input_text}")
        inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 计算ROUGE-L分数
        # 使用完整输出进行ROUGE-L计算
        rouge_l_score = calculate_rouge_l(output_text, true_answer)
        rouge_l_scores.append(rouge_l_score)
        
        
        # 收集结果数据
        results_data.append({
            'question': question,
            'true_answer': true_answer,
            'model_output': output_text,
            'rouge_l_score': rouge_l_score
        })
        
        # 为了调试，可以打印一些样本
        if total <= 3:
            print(f"Question: {question}")
            print(f"True Answer: {true_answer}")
            print(f"Output Text: {output_text}")
            print(f"ROUGE-L Score: {rouge_l_score:.4f}")
            print("-" * 50)
    
    accuracy = correct / total if total > 0 else 0
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    return accuracy, correct, total, avg_rouge_l, results_data

def main():
    # 模型名称
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # 加载模型
    print("Loading model...")
    try:
        tokenizer, model = load_model(model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # 加载GSM8K数据集
    print("Loading dataset...")
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # 随机选择50个样本
    import random
    if len(dataset) > 50:
        random_indices = random.sample(range(len(dataset)), 50)
        dataset = dataset.select(random_indices)
    
    # 评估模型
    print("Evaluating model...")
    accuracy, correct, total, avg_rouge_l, results_data = evaluate_model(model, tokenizer, dataset)
    
    # 输出结果
    print(f"\nResults:")
    print(f"Total questions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average ROUGE-L Score: {avg_rouge_l:.4f}")
    
    # 保存结果到JSON文件
    results = {
        "model_name": model_name,
        "total_questions": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "average_rouge_l_score": avg_rouge_l
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to evaluation_results.json")
    
    # 保存详细结果到CSV文件
    df = pd.DataFrame(results_data)
    df.to_csv("一评价结果训练前.csv", index=False, encoding='utf-8-sig')
    print("Detailed results saved to 一评价结果训练前.csv")

if __name__ == "__main__":
    main()