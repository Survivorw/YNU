import json

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, \
    TextDataset, AutoModel, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).quantize(4).cuda()

max_input_length = 128
max_target_length = 128


def preprocess_function(examples):
    inputs = [src for src in examples['label']]
    targets = [tgt for tgt in examples['text']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,
                             padding=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True,
                           padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model
)

train_dataset = load_dataset("json", data_files="Data/mytrain.json")
valid_dataset = load_dataset("json", data_files="Data/myvalid.json")
tokenized_train_dataset = train_dataset["train"].map(preprocess_function, batched=True)
tokenized_valid_dataset = valid_dataset["train"].map(preprocess_function, batched=True)
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    learning_rate=2e-6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy="epoch",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
)
trainer.train()
trainer.save_model()
