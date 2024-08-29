from flask import Flask, request, jsonify, render_template
from test import text

app = Flask(__name__)


# 模拟一个简单的文本生成函数
def generate_text(input_text):
    # 这里可以填入你的生成文本逻辑，例如调用本地的语言模型
    # 此处暂时返回一个固定的文本
    return text(input_text)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_text = data.get('input')

    if input_text:
        generated_text = generate_text(input_text)
        return jsonify({'result': generated_text})
    else:
        return jsonify({'error': '无法生成文本。'})


@app.route('/')
def index():
    return render_template('zw (2).html')


if __name__ == '__main__':
    app.run(debug=True)
