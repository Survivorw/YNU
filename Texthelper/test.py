from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).quantize(4).cuda()


def text(inputs):
    response, history = model.chat(tokenizer, "生成作文只给出作文，不要有其他的话", history=[])
    text = inputs
    response, history = model.chat(tokenizer, text, history=history)
    return response
