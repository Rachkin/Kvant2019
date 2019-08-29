window.onload = function () {
  var socket = new WebSocket("ws://localhost:8881");

  socket.onopen = function() {
    alert("Connection sucessful.");
  };

  socket.onclose = function(event) {
    if (event.wasClean) {
    alert('Connection closed clear');
    } else {
    alert('Connection cutted'); // например, "убит" процесс сервера
    }
    console.log('Code: ' + event.code + ' reason: ' + event.reason);
  };

  socket.onmessage = function(event) {
    //document.getElementById("content").innerHTML = "Your emotion is " + event.data;
    alert("Your emotion is " + event.data);
  };

  socket.onerror = function(error) {
    console.log("Error " + error.message);
  };
/////////////////////////////////////////////////////////////////////
  var canvas = document.getElementById('canvas');
  var video = document.getElementById('video');
  var button = document.getElementById('button');
  var allow = document.getElementById('allow');
  var context = canvas.getContext('2d');
  var videoStreamUrl = false;

  var img = new Image();
  img.src = "sfu_logo.png";
  context.drawImage(img, 0, 0);


  // функция которая будет выполнена при нажатии на кнопку захвата кадра
  var captureMe = function () {
    if (!videoStreamUrl) alert('Something wrong1');
    // переворачиваем canvas зеркально по горизонтали (см. описание внизу статьи)
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    // отрисовываем на канвасе текущий кадр видео
    context.drawImage(video, 0, 0, video.width, video.height);
    // получаем data: url изображения c canvas
    var base64dataUrl = canvas.toDataURL('image/png');
    context.setTransform(1, 0, 0, 1, 0, 0); // убираем все кастомные трансформации canvas
    // на этом этапе можно спокойно отправить  base64dataUrl на сервер и сохранить его там как файл (ну или типа того)
    // но мы добавим эти тестовые снимки в наш пример:
    var img = new Image();
    img.src = base64dataUrl;
    //window.document.body.appendChild(img);
//socket.send(video);

    img.style.width = "48px";
    img.style.height = "48px";
    cnv = new Canvas();
    ctx = new Context();

    ctx.drawImage(img, 0, 0);

    array = ctx.getImageData(0, 0, 48, 48).data
    var r = 0, g = 0, b = 0;
    var grey = [];
    for(var i = 0; i < array.size(); i++){
    if(i % 4 == 0){
      r = array[i];
    }if(i % 4 == 1){
      g = array[i];
    }if(i % 4 == 2){
      b = array[i];
    }if(i % 4 == 3){
      grey.push((0.2125 * r) + (0.7154 * g) + (0.0721 * b));
    }
    }

    socket.send(grey);

  }

  button.addEventListener('click', captureMe);

    // navigator.getUserMedia  и   window.URL.createObjectURL (смутные времена браузерных противоречий 2012)
  navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
  window.URL.createObjectURL = window.URL.createObjectURL || window.URL.webkitCreateObjectURL || window.URL.mozCreateObjectURL || window.URL.msCreateObjectURL;

    // запрашиваем разрешение на доступ к поточному видео камеры
  navigator.getUserMedia({video: true}, function (stream) {
      // разрешение от пользователя получено
      // скрываем подсказку
    //  allow.style.display = "none";
      // получаем url поточного видео
    videoStreamUrl = window.URL.createObjectURL(stream);
      // устанавливаем как источник для video
    video.src = videoStreamUrl;
  }, function () {
    console.log('Something wrong :P');
  });
};
