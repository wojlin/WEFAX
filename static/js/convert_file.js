function convert_file()
{
     const file_info = document.getElementById("file_info");
     const content = document.getElementById("content");
     const content_actual = document.getElementById("content-actual");
     const content_title = document.getElementById("content-title");
     file_info.classList.add("slide_out");
     content.style.display = 'block';


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
                if(data["data_type"] == "progress_bar")
                {

                    if(content_actual.innerHTML == 'empty')
                    {
                        content.classList.add("slide_in");
                        content_actual.innerHTML = data["progress_title"];
                        content_title.innerHTML = data["progress_title"];
                        renderProgress(data["percentage"]);
                    }
                    else if(content_actual.innerHTML != data["progress_title"])
                    {
                        content_actual.innerHTML = data["progress_title"];
                        content.classList.remove("slide_in");
                        content.classList.add("slide_out");
                        setTimeout(function(){
                            content.classList.remove("slide_out");
                            content.classList.add("slide_in");
                            content_title.innerHTML = data["progress_title"];
                            renderProgress(data["percentage"]);
                        }, 1000);

                    }else
                    {
                        content_actual.innerHTML = data["progress_title"];
                        content_title.innerHTML = data["progress_title"];
                        renderProgress(data["percentage"]);
                    }



                }
                else
                {
                    console.log(data);
                }

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