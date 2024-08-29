<script setup lang="ts">
</script>

<template>
  <div class="wrapper">
    <div class="header">
      <div class="title">Cesium</div>
    </div>
    <div class="body">
      <div class="first">
        <div class="tool">
          
          <button>
            <img src="https://ts2.cn.mm.bing.net/th?id=OIP-C.7JUyymzrcpiIlzDs9h2XHAAAAA&w=176&h=176&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2"/>
            测高
          </button>
        </div>
        <div class="tool">
          <button>
            <img src="https://ts2.cn.mm.bing.net/th?id=OIP-C.7JUyymzrcpiIlzDs9h2XHAAAAA&w=176&h=176&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2"/>
            测宽
          </button>
        </div>
        <div class="tool">
          <button>
            <img src="https://img-qn.51miz.com/Element/00/78/51/97/c51922d9_E785197_212875a3.png"/>
            路况
          </button>
        </div>
      </div>
      <div class="second">
        <img src="">
      </div>
      <div class="third">
        <label for="file">分析进度:</label>
        <progress id="file" max="100" value="70">70%</progress>
        <div class="btn">
          <h1>淹没速度 m/s</h1>
          <input>
          <button>淹没分析</button>
        </div>
        <div class="btn1">
          <button>北辰路</button>
        </div>
        <div class="btn1">
          <button>广福路</button>
        </div>
      </div>
    </div>
    <div class="footer">
      <div class="info">
        <div class="city">
          <img src="https://tse3-mm.cn.bing.net/th/id/OIP-C.OIHmKm6msj50ll6xqhGADAAAAA?w=200&h=201&c=7&r=0&o=5&dpr=1.3&pid=1.7"/>
          昆明
        </div>
        <div class="weather">
          <img src="data:image/jpeg;base64,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"/>
          晴 7/20°C
        </div>
    </div>
    </div>
  </div>
</template>

<style scoped>
.wrapper {
  height: 100%;
  width: 100%;
  background: linear-gradient(70deg, #c8ddff, #cae9f8);
}
.wrapper .header {
  height: 7vw;
  font-family:Arial,sans-serif;
  display: flex;
  align-items: center;
  justify-content: center;
}

.wrapper .header .title{
  height: 5vw;
  font-family:Arial,sans-serif;
  display: flex;
  align-items: center;
  color: #5f6477;
  background:#FFFFFF;
  font-size: 4vw;
  border-radius: 1vw;
}
.wrapper .body {
  width: 100%;
  height: 34vw;
  display: flex;
  
}
.wrapper .body .first {
  width: 8%;
  height: 15vw;
  margin-top: 15vw;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.wrapper .body .first .tool img{
  height: 1vw;
}
.wrapper .body .first button {
  height: 3vw;
  width: 4vw;
  margin-left: 1vw;
  font-size: 1vw;
  display: flex;
  background:#FFFFFF;
  border-radius: 1vw;
  justify-content: space-around;
  align-items: center;
  font-family:Arial,sans-serif;
  color: #5f6477;
  border:none;
  cursor:pointer;
}
.wrapper .body .second {
  width: 80%;
}
.wrapper .body .third {
  width: 10%;
  margin-top: 10vw;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  margin-right: 1vw;
}

.wrapper .body .third .btn {
  height: 5vw;
  border-radius: 1vw;
  background:#FFFFFF;
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  align-items: center;
  margin-left: 3vw;
  width: 8vw;
}
.wrapper .body .third .btn h1 {
  font-size: 0.8vw;
  font-family:Arial,sans-serif;
  color: #5f6477;
}
.wrapper .body .third .btn input {
  width: 6vw;
  border-radius: 1vw;
  
  background:#FFFFFF;
}

.wrapper .body .third .btn button {
  background:#FFFFFF;
  border-radius: 1vw;
  font-family:Arial,sans-serif;
  color: #5f6477;
  cursor:pointer;
}

.wrapper .body .third .btn1 button { 
  margin-left: 3vw;
  width: 8vw;
  background:#FFFFFF;
  border-radius: 1vw;
  font-size: 1.5vw;
  height: 4vw;
  color: #5f6477;
  border:none;
  cursor:pointer;
}
.wrapper .footer {
  height: 5.5vw;
  width: 100%;
  margin-top: 2vw;
  
}
.wrapper .footer .info{
  height: 3vw;
  width: 50vw;
  border-radius: 1vw;
  margin-top: 2vw;
  display: flex;
  align-items: center;
  justify-content:space-around;
  font-family:Arial,sans-serif;
  color: #5f6477;
  background:#FFFFFF;
  margin-left: 25vw;
}
.wrapper .footer .info .city {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: space-around;
  width: 5.5vw;
  font-size: 1.3vw;
}
.wrapper .footer .info .city img{
  height: 2.2vw;
}
.wrapper .footer .info .weather {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: space-around;
  width: 9vw;
  font-size: 1.3vw;
}
.wrapper .footer .info .weather img{
  height: 2.2vw;
}
</style>