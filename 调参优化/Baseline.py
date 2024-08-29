import numpy as np
import time
import GPUtil
import logging
import os
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
from peft import get_peft_model,TaskType,LoraConfig

if not os.path.exists("result"):
    os.makedirs("result")
logging.basicConfig(filename='./result/training.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=['k_proj','v_proj']
)

model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model=get_peft_model(model,peft_config)
model.print_trainable_parameters()

metric = evaluate.load("sacrebleu")
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "zh"
dataset_dict = load_dataset("opus100", "en-zh", split=["train", "validation", "test"])
train_dataset, valid_dataset, test_dataset = dataset_dict

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {
        'bleu-1': result['precisions'][0],
        'bleu-2': result['precisions'][1],
        'bleu-3': result['precisions'][2],
        'bleu-4': result['precisions'][3],
    }


tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"./result/{model_name}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=1,
    predict_with_generate=True,
    eval_steps=5000,
    metric_for_best_model="bleu-4"
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
start_time = time.time()
gpus = GPUtil.getGPUs()
gpu_memory_start = gpus[0].memoryFree

trainer.train()  # 模型训练

gpus = GPUtil.getGPUs()
gpu_memory_end = gpus[0].memoryFree
end_time = time.time()
training_time = end_time - start_time
logger.info(f"Total training time: {training_time} seconds")
gpu_memory_used = gpu_memory_start - gpu_memory_end
logger.info(f"training GPU memory used: {gpu_memory_used} MB")


logger.info(trainer.predict(tokenized_test_dataset).metrics)

