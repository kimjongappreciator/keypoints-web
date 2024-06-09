const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const socket = io("ws://localhost:5000"); // Use io instead of new WebSocket

var frames = [];

var startTime;
var endTime;

socket.on("connect", function() {
  console.log("Conectado al servidor");
});

socket.on("prediction", function(data) {
  const prediction = data.prediction;
  endTime = performance.now();
  console.log("Predicción:", prediction);
  console.log("Tiempo de ejecución:", (endTime - startTime)/1000);
  // Update UI with prediction
});

navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
  video.srcObject = stream;
  startTime = performance.now();
  setInterval(function() {
    ctx.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg");
    //frames.push(imageData);
    
    socket.emit("message", {"frames": imageData}); // Use emit instead of send

    /*if (frames.length === 30) {
      console.log("enviando frames: " + frames.length)
      //socket.emit("message", {"frames": frames}); // Use emit instead of send
      socket.emit("message", "hola");
      frames = [];
    }*/
  }, 100);
});