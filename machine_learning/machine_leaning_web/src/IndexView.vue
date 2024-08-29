<template>
  <body>
    <div class="above-part">
      <ul>
        <li v-for="item in btns">
          <button @click="plot(item.name)" value="bear">{{ item.name }}</button>
        </li>
      </ul>
    </div>

    <div class="modal" v-if="isModalOpen">
      <div class="modal-content">
        <span class="close" @click="closeModal"></span>
        <div class="btn-container">
          <button id='left' @click="left">
          <span class="btn-title">&lt</span>
        </button>
        </div>
        
        <img id='image' style="height: 20vw;width: 30vw;" :src="imageUrl2" alt="图片">
        
        <div class="btn-container">
          <button id='right' @click="right">
          <span class="btn-title">></span>
        </button>

        </div>
        <button id="cancel" @click="cancel">Cancel</button>
      </div>
      <div class="black_overlay"></div>
    </div>
    <div class="under-part">
      <div class="load">
        <img id='image' style="height: 300px;width: 400px;" :src="imageUrl">
        <div class="card_content">
          <p class="card_title">柱状图</p>
          <p class="card_description">对上传的图像利用训练好的CNN模型进行识别，并给出10-类上的预测概率分布柱状图示。</p>
        </div>
      </div>

      <div id="mnist-pad">
        <input type="color" v-model="selectedColor">
        <div class="mnist-pad-body">
          <canvas ref='canvas' @mousedown="startDrawing" @mousemove="draw" @mouseup="stopDrawing"></canvas>
          <div class="board"></div>
        </div>

        <div class="mnist-pad-footer">
          <div class="mnist-pad-result">

          </div>
          <div class="mnist-pad-actions">
            <button type="button" id="mnist-pad-clear" class="cssbuttons-io-button" @click="clearCanvas">
              <img src="./assets/清空 (1).svg">
              <span>清空</span>
            </button>
            <button type="submit" id="mnist-pad-save" class="cssbuttons-io-button" @click='uploadCanvasAsImage'>

              <img src="./assets/上传.svg">
              <span>上传</span>
            </button>
          </div>
        </div>

      </div>
    </div>
  </body>
</template>

<script setup>
import axios from "axios";
import { onMounted, ref } from "vue";

const imageUrl = ref('')
const imageUrl2 = ref('')
const canvas = ref(null);
const ctx = ref(null);
const isDrawing = ref(false);
const inputValue = ref('');
const selectedColor = ref('#000000')
const lastX = ref(0);
const lastY = ref(0);
const isModalOpen = ref(false)
const btns = ref([
  {
    name: 'apple',
  },
  {
    name: 'bird',
  },
  {
    name: 'cat',
  },
  {
    name: 'bicycle',
  },
  {
    name: 'bus',
  },
  {
    name: 'ambulance',
  },
  {
    name: 'foot',
  },
  {
    name: 'pig',
  },
  {
    name: 'owl',
  },
  {
    name: 'bear',
  }
])
const pictures = ref([])

const startDrawing = (e) => {
  isDrawing.value = true;
  [lastX.value, lastY.value] = [e.offsetX, e.offsetY];
};

const draw = (e) => {
  if (!isDrawing.value) return;
  ctx.value.strokeStyle = selectedColor.value;
  ctx.value.lineCap = 'round';
  ctx.value.lineWidth = 5;

  ctx.value.beginPath();
  ctx.value.moveTo(lastX.value, lastY.value);
  ctx.value.lineTo(e.offsetX, e.offsetY);
  ctx.value.stroke();

  [lastX.value, lastY.value] = [e.offsetX, e.offsetY];
};

const stopDrawing = () => {
  isDrawing.value = false;
};

onMounted(() => {
  const canvasElement = canvas.value;
  ctx.value = canvasElement.getContext('2d');
  canvasElement.width = 200;
  canvasElement.height = 200;
  ctx.value.fillStyle = 'white';
  ctx.value.fillRect(0, 0, canvasElement.width, canvasElement.height);
})

const clearCanvas = () => {
  const canvasElement = canvas.value;
  ctx.value.clearRect(0, 0, canvasElement.width, canvasElement.height);
  ctx.value.fillStyle = 'white';
  ctx.value.fillRect(0, 0, canvasElement.width, canvasElement.height);
};



const uploadCanvasAsImage = () => {
  const dataUrl = canvas.value.toDataURL('image/jpeg');
  axios.post('http://localhost:5000/predict', {
    imageUrl: dataUrl,
  }, {
    headers: {
      'Content-Type': 'application/text'
    }
  }).then(response => {
    console.log(response);
    imageUrl.value = response.data.result
  }).catch(error => {
    console.error(error);
  });
};

function plot(name) {
  axios.get('http://localhost:5000/plot', {
    params: { name: name }
  }).then((response) => {
    console.log(response);
    pictures.value = response.data.result
    imageUrl2.value = pictures.value[0]
    isModalOpen.value = !isModalOpen.value
  }).catch(error => {
    console.error(error);
  });
}


function left() {
  let index = pictures.value.indexOf(imageUrl2.value) + 1
  if (index == pictures.value.length) {
    imageUrl2.value = pictures.value[0]
  } else {
    imageUrl2.value = pictures.value[index]
  }
}

function right() {
  let index = pictures.value.indexOf(imageUrl2.value) - 1
  if (index == pictures.value.length) {
    imageUrl2.value = pictures.value[0]
  } else {
    imageUrl2.value = pictures.value[index]
  }
}

function cancel() {
  isModalOpen.value = !isModalOpen.value
}
</script>

<style scoped>
body {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
  user-select: none;
  margin: 0;
  padding: 0;
  background-image: url(./assets/speedsketch-background.svg);
  background-size: 90%;
}

.modal {
  height: 100%;
  width: 100%;
  margin-top: -9.5vw;
  display: flex;
  justify-content: center;
  align-items: center;
  position: absolute;
  top: 20%;
  border: 1px solid #bbbbbb;
  border-radius: 20px;
  background-color: white;
  z-index: 1002;
  /*层级要比.black_overlay高，这样才能显示在它前面*/
  overflow: auto;
  z-index: 1002;
  opacity: .80;

}

.modal img {
  opacity: 1;
}

.above-part ul {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
}

.above-part .gallary {
  display: flex;
  width: 5vw;
  height: 5vw;
}
.modal-content{
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
}
.under-part {
  display: flex;
  justify-content: space-around;
  align-items: center;
  height: 88.6vh;
  width: 100%;
  user-select: none;
  margin: 0;
  padding: 0;

}

.load {
  margin-left: 15vw;
  position: relative;
  width: 400px;
  height: 300px;
  background-color: #f2f2f2;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  perspective: 1000px;
  box-shadow: 0 0 0 5px #ffffff80;
  transition: all 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  margin-bottom: 10vw;
}

.load img {
  fill: #333;
  transition: all 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.load:hover {
  transform: scale(1.05);
  box-shadow: 0 8px 16px rgba(255, 255, 255, 0.2);
}

.load .card_content {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  padding: 20px;
  box-sizing: border-box;
  background-color: #f2f2f2;
  transform: rotateX(-90deg);
  transform-origin: bottom;
  transition: all 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.load:hover .card_content {
  transform: rotateX(0deg);
}

.card_title {
  margin: 0;
  font-size: 24px;
  color: #333;
  font-weight: 700;
}

.load:hover img {
  scale: 0;
}

.card_description {
  margin: 10px 0 0;
  font-size: 14px;
  color: #777;
  line-height: 1.4;
}

#mnist-pad {
  position: relative;
  display: flex;
  flex-direction: column;
  font-size: 1em;
  width: 30%;
  height: 29%;
  margin-right: 10vw;
  margin-bottom: 10vw;
  padding: 16px;
}

.mnist-pad-body canvas {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  border-radius: 4px;
  box-shadow: 0 0 5px rgba(0, 0, 0, 0.02) inset;
}

.mnist-pad-footer {
  color: #C3C3C3;
  font-size: 1.2em;
  height: 13%;
  width: 30vw;
  margin-top: 220px;

}

#mnist-pad input {
  margin-left: 50px;

}

.mnist-pad-actions {
  display: flex;
  width: 20vw;
  justify-content: space-around;
  margin-left: 5vw;
}
.btn-container {
  height: 3vw;
  width: 8vw;
  display: flex;
  justify-content: center;
  align-items: center;
  --color-text: #ffffff;
  --color-background: #725AC1;
  --color-outline: #725AC1;
  --color-shadow: #725AC1;
}
#left {
  display: inline-block;

  padding: 5px 15px;
  text-align: center;
  font-size: 18px;
  letter-spacing: 1px;
  text-decoration: none;
  color: #725AC1;
  background: #eeeeee;
  cursor: pointer;
  transition: ease-out 0.5s;
  border: 2px solid #725AC1;
  border-radius: 10px;
  box-shadow: inset 0 0 0 0 #725AC1;

}

#right {
  display: inline-block;
  padding: 5px 15px;
  text-align: center;
  font-size: 18px;
  letter-spacing: 1px;
  text-decoration: none;
  color: #725AC1;
  background: #eeeeee;
  cursor: pointer;
  transition: ease-out 0.5s;
  border: 2px solid #725AC1;
  border-radius: 10px;
  box-shadow: inset 0 0 0 0 #725AC1;

}
#left:hover {
  color: white;
  box-shadow: inset 0 -100px 0 0 #725AC1;
}

#left:active {
  transform: scale(0.9);
}
#right:hover {
  color: white;
  box-shadow: inset 0 -100px 0 0 #725AC1;
}

#right:active {
  transform: scale(0.9);
}
#cancel{
  position: relative;
  display: inline-block;
  padding: 10px 25px;
  text-align: center;
  font-size: 15px;
  letter-spacing: 1px;
  text-decoration: none;
  color: #725AC1;
  background: #eeeeee;
  cursor: pointer;
  transition: ease-out 0.5s;
  border: 2px solid #725AC1;
  border-radius: 10px;
  box-shadow: inset 0 0 0 0 #725AC1;
  margin-bottom: 25vw;
}
#cancel:hover {
  color: white;
  box-shadow: inset 0 -100px 0 0 #725AC1;
}

#cancel:active {
  transform: scale(0.9);
}
.above-part .modal {
  display: flex;
  justify-content: center;
  align-items: center;
}

.above-part button {
  position: relative;
  display: inline-block;
  margin: 15px;
  padding: 15px 30px;
  text-align: center;
  font-size: 18px;
  letter-spacing: 1px;
  text-decoration: none;
  color: #725AC1;
  background: #eeeeee;
  cursor: pointer;
  transition: ease-out 0.5s;
  border: 2px solid #725AC1;
  border-radius: 10px;
  box-shadow: inset 0 0 0 0 #725AC1;
}

.above-part button:hover {
  color: white;
  box-shadow: inset 0 -100px 0 0 #725AC1;
}

.above-part button:active {
  transform: scale(0.9);
}

.cssbuttons-io-button {
  display: flex;
  align-items: center;
  font-family: inherit;
  font-weight: 500;
  font-size: 8px;
  padding: 0.8em 1.5em 0.8em 1.2em;
  color: white;
  background: #ad5389;
  background: linear-gradient(0deg, rgb(120, 47, 255) 0%, rgb(185, 132, 255) 100%);
  border: none;
  box-shadow: 0 0.7em 1.5em -0.5em rgb(184, 146, 255);
  letter-spacing: 0.05em;
  border-radius: 20em;
}

.cssbuttons-io-button img {
  margin-right: 5px;
  width: 1.2vw;
  height: 1.2vw;
}

.cssbuttons-io-button span {
  font-size: 1vw;
}

.cssbuttons-io-button:hover {
  box-shadow: 0 0.5em 1.5em -0.5em rgb(149, 91, 255);
}

.cssbuttons-io-button:active {
  box-shadow: 0 0.3em 1em -0.5em rgb(160, 109, 255);
}

.mnist-pad-body .board {
  display: flex;
  height: 12px;
  background-color: #999;
  border-radius: 0 0 12px 12px;
  bottom: 0;
  box-shadow: 15px 15px 0 0 rgba(0, 0, 0, .2);
  position: absolute;
  margin-left: 125px;
  width: 210px;
}
</style>