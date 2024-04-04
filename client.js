const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const socket = new WebSocket("ws://localhost:5000");

var frames = [];

socket.onopen = function() {
    console.log("Conectado al servidor");
};

socket.onmessage = function(event) {
    const prediction = JSON.parse(event.data).prediction;    
    console.log("Predicción:", prediction);    
};

navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
    video.srcObject = stream;
    setInterval(function() {
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL("image/jpeg");
        frames.push(imageData);

        if (frames.length === 30) {
            res = socket.send(JSON.stringify({"frames": frames}));
            console.log(res);
            frames = [];
        }
    }, 100);
});