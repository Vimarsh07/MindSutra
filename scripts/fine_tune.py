# fine_tune.py
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# Load dataset from merged CSV
merged_data_path = '/Users/vimarsh/Desktop/MindMate/data/merged_df.csv'
merged_df = pd.read_csv(merged_data_path)

# Remove rows with missing prompt or response
merged_df.dropna(subset=['prompt', 'response'], inplace=True)

# Load tokenizer and model
pretrained_model_name = "vibhorag101/llama-2-7b-chat-hf-phr_mental_therapy"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

# Convert DataFrame to Dataset for use with transformers
huggingface_dataset = Dataset.from_pandas(merged_df)

def format_prompt_response(examples):
    # Concatenate prompt and response with an EOS token
    return {'text': [f"{prompt}{tokenizer.eos_token}{response}" for prompt, response in zip(examples['prompt'], examples['response'])]}

# Tokenize dataset
formatted_dataset = huggingface_dataset.map(format_prompt_response, batched=True)

# Define training parameters
training_params = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,  # No use of fp16 as specified
    gradient_accumulation_steps=1,
)

# Update model configuration for LoRA and 4-bit optimizations
model.config.update({
    "lora_r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "use_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "use_nested_quant": False
})

# Set up Trainer
model_trainer = Trainer(
    model=model,
    args=training_params,
    train_dataset=formatted_dataset,
    eval_dataset=formatted_dataset,
)

# Begin training process
model_trainer.train()

# Save fine-tuned model
model_trainer.save_model("/Users/Vimarsh/Desktop/MindSutra/models")
