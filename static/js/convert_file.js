function convert_file()
{
     const file_info = document.getElementById("file_info");
     const content = document.getElementById("content");
     file_info.classList.add("slide_out");
     content.style.display = 'block';
     content.classList.add("slide_in");

     var data = new FormData();

     data.append('datestamp', document.getElementById("file_info_datestamp").innerHTML);


     var socket = io();

     socket.addEventListener('message', function (event) {
                console.log('Message from server ', event.data);
            });
            socket.on('connect', function() {
                socket.emit('get_progress', {data: document.getElementById("file_info_datestamp").innerHTML});
            });

            socket.on('upload_progress', function(data) {
                console.log(data)
            });





    for (const value of data.values()) {
      console.log(value);
    }
    var req = fetch('/convert_file', {
      method: 'POST',
      body: data
    });

    req.then(function(response) {
      console.log('success');
      response.text().then(function (text) {
          //console.log(text);

        });

    }, function(error) {
      console.error('failed due to network error or cross domain')
    })

}