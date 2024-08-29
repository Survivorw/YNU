import ktrain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
src_text=input()
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

predictor = ktrain.load_predictor('models')
message = res
prediction = predictor.predict(message)

print(prediction)