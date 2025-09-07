import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import deepspeed

# 加载模型和分词器
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Changed from torch.bfloat16 to torch.float16 for better CUDA compatibility
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return tokenizer, model

# 准备数据集
def prepare_dataset(tokenizer, dataset):
    def tokenize_function(examples):
        inputs = [f"Question: {q}\nAnswer: Let's think step by step. {a}" for q, a in zip(examples['question'], examples['answer'])]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# 主函数
def main():
    # 模型名称
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # 加载模型
    print("Loading model...")
    tokenizer, model = load_model(model_name)
    
    # 加载GSM8K数据集
    print("Loading dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # 准备数据集
    print("Preparing dataset...")
    tokenized_dataset = prepare_dataset(tokenizer, dataset)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,
        dataloader_pin_memory=False,
        #deepspeed="./ds_config.json"
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    trainer.save_model("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")
    
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()