from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from huggingface_hub import HfApi
from datasets import load_dataset
import requests
import os
import torch

os.environ['http_proxy'] = 'http://dalian-webproxy.openjawtech.com:3128'
os.environ['https_proxy'] = 'http://dalian-webproxy.openjawtech.com:3128'

model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cpu")
base_model.to(device)


def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer


# ### Try the base model
dataset_path = "lamini/taylor_swift"
finetuning_dataset = load_dataset(dataset_path)
test_sample = finetuning_dataset["test"][0]
test_text = test_sample['question']
print("Question input (test):", test_text)
print(f"Correct answer from Lamini docs: {test_sample['answer']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))

# 应用转换函数到训练集
#train_dataset = finetuning_dataset['train'].map(convert_to_int64)
total_size = len(finetuning_dataset['train'])
train_size = int(0.8 * total_size)
#val_size = int(0.2 * total_size)
#test_size = total_size - train_size - val_size

#train_dataset = finetuning_dataset['train'].select(range(train_size))
train_dataset = finetuning_dataset['train']
# 设置训练参数
max_steps = 3000
num_train_epochs = 3
trained_model_name = f"lamini_docs_{num_train_epochs}_epochs"
output_dir = trained_model_name

training_args = TrainingArguments(
    #max_steps=max_steps,
    learning_rate=1.0e-5,
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

use_hf = True
training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length": 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

model_flops = (
  base_model.floating_point_ops(
    {
       "input_ids": torch.zeros(
           (1, training_config["model"]["max_length"])
      )
    }
  )
  * training_args.gradient_accumulation_steps
)


print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
# 创建 Trainer
trainer = Trainer(
    model=base_model,
    args=training_args,
    #model_flops=model_flops,
    train_dataset=train_dataset,
    eval_dataset=finetuning_dataset["test"],
)

# 开始训练
trainer.train()
# 生成文本
test_sample = finetuning_dataset["test"][0]
test_text = test_sample['question']
print("Question input (test):", test_text)
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))
