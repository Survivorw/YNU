from flask import  Flask,render_template,url_for,request,jsonify
import re
import base64
import io
from PIL import Image
import numpy as np
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import os
from flask_cors import CORS
import json
app = Flask(__name__)


@app.route('/')
def index():    
    return render_template("HandWrite.html")


def predict():    
    data = request.get_data()
    data = json.loads(data)['imageUrl']
    data=data.replace('data:image/jpeg;base64,', '')
    data = io.BytesIO(base64.b64decode(data))
    image= Image.open(data)
    image=image.resize((28, 28))
    npy=np.array(image)
    input_data=npy.reshape(-1,28,28,1)
    model=load_model('cnn.h5')
    y_pred=model.predict(input_data)
    plt.clf()
    plt.bar(np.arange(10), y_pred[0])
    plt.xticks(np.arange(10),['ambulance','apple','bear','bicycle','bird','bus','cat','foot','owl','pig'])
    plt.xlabel('Class')
    plt.ylabel('Probability')
    label_names=['ambulance','apple','bear','bicycle','bird','bus','cat','foot','owl','pig']
    plt.title(f'Predicted class: {label_names[np.argmax(y_pred[0])]} ')
    img = BytesIO()
    plt.savefig(img,format='png')
    img.seek(0)
    return img

@app.route('/predict',methods=['post'])

def plot():
    img=predict()
    plot_data = img.getvalue()
    imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    return jsonify({'result':imd})

@app.route('/plot',methods=['get'])
def plotimage():
    name = request.args.get('name')
    image_data=[]
    path = os.path.join('data2/', name)
    for image_path in os.listdir(path):
        image_path=os.path.join(path,image_path)
        data_all = np.load(image_path)
        for i in range(10):
            data = data_all[i, :] 
            data = data.reshape(28, 28)
            img = Image.fromarray(data) 
            image=BytesIO()
            img.save(image,'png')
            image.seek(0)
            image=image.getvalue()
            image=base64.b64encode(image)
            image=image.decode()
            image="data:image/png;base64," +image
            image_data.append(image)
    return jsonify({'result':image_data})

   
if __name__=="__main__":
    app.run()