//selecting all required elements
const upload_panel = document.getElementById("upload");
const file_info = document.getElementById("file_info");
const dropArea = document.querySelector(".drag-area"),
dragText = dropArea.querySelector("header"),
button = dropArea.querySelector("button"),
input = dropArea.querySelector("input");
let file; //this is a global variable and we'll use it inside multiple functions

function upload(file)
{
    console.log('file upload');

  var data = new FormData()
   data.append('file', file[0]);


    for (const value of data.values()) {
  console.log(value);
}
    var req = fetch('/load_file', {
      method: 'POST',
      body: data
    });

    req.then(function(response) {
      console.log('success');
      response.text().then(function (text) {
          console.log(text);
            var json = JSON.parse(text);
            document.getElementById('file_info_datestamp').innerHTML = json['datestamp'];
             document.getElementById('file_info_port').innerHTML = json['tcp_port'];
            document.getElementById('file_info_filename').innerHTML = json['filename'];
            document.getElementById('file_info_channels').innerHTML = json['channels'];
            document.getElementById('file_info_length').innerHTML = parseFloat(json['length']).toFixed(2) + ' s';
            document.getElementById('file_info_rate').innerHTML = parseFloat(json['sample_rate'])/1000 + ' KHz';
            upload_panel.classList.add("slide_out");
            file_info.style.display = 'block';
            file_info.classList.add("slide_in");


            let socket = new WebSocket("wss://localhost:"+json['tcp_port']);

            socket.onopen = function(e) {
              console.log("[open] Connection established");
            };

            socket.onmessage = function(event) {
              console.log("[message] Data received from server:" + event.data);
            };

            socket.onclose = function(event) {
              if (event.wasClean) {
                console.log("[close] Connection closed cleanly, code=" + event.code + " reason=" + event.reason);
              } else {
                // e.g. server process killed or network down
                // event.code is usually 1006 in this case
                console.log('[close] Connection died');
              }
            };

            socket.onerror = function(error) {
              console.log("[error]" +error.message);
            };

        });

    }, function(error) {
      console.error('failed due to network error or cross domain')
    })
}

button.onclick = ()=>{
  input.click(); //if user click on the button then the input also clicked
}
input.addEventListener("change", function(){
  //getting user select file and [0] this means if user select multiple files then we'll select only the first one
  file = this.files;
  dropArea.classList.add("active");
    upload(file);
});
//If user Drag File Over DropArea
dropArea.addEventListener("dragover", (event)=>{
  event.preventDefault(); //preventing from default behaviour
  dropArea.classList.remove("notactive");
  dropArea.classList.add("active");

  dragText.textContent = "Release to Upload File";
});
//If user leave dragged File from DropArea
dropArea.addEventListener("dragleave", ()=>{
  dropArea.classList.remove("active");
  dropArea.classList.add("notactive");
  dragText.textContent = "Drag & Drop to Upload File";
});
//If user drop File on DropArea
dropArea.addEventListener("drop", (event)=>{
  event.preventDefault(); //preventing from default behaviour
  //getting user select file and [0] this means if user select multiple files then we'll select only the first one
  file = event.dataTransfer.files;
  dropArea.classList.remove("notactive");
  dropArea.classList.add("active");
    upload(file);

});