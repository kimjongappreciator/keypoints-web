const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const socket = io("ws://192.168.1.38:5000"); // Use io instead of new WebSocket

var frames = [];

var startTime;
var endTime;

socket.on("connect", function() {
  console.log("Conectado al servidor");
});

socket.on("prediction", function(data) {
  const prediction = data.prediction;
  //endTime = performance.now();
  console.log("Predicción:", prediction);
  //console.log("Tiempo de ejecución:", (endTime - startTime)/1000);
  
});

navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
  video.srcObject = stream;
  //startTime = performance.now();
  setInterval(function() {
    ctx.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg");
    
    socket.emit("message", {"frames": imageData}); // Use emit instead of send

    
  }, 100);
});