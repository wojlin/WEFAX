function get_audio_devices()
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function()
    {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
        {
            var json = JSON.parse(xmlHttp.responseText);
            console.log(json);
            var audio_devices_select = document.getElementById("audio_devices_select");

            for(var i in json)
            {
                var key = i;
                var name = json[i]["name"];
                var index = json[i]["index"];
                console.log(name);
                var option = document.createElement("option");
                option.text = name;
                option.value = index;
                audio_devices_select.appendChild(option);
            }

            change_audio_device(document.getElementById("audio_devices_select"));

        }
    }
    xmlHttp.open("GET", '/get_audio_devices', true); // true for asynchronous
    xmlHttp.send(null);
}

function change_audio_device(element)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function()
    {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
        {
            var text = xmlHttp.responseText;
            console.log(text);

        }
    }
    var val = element.value;

    xmlHttp.open("GET", '/change_audio_device/'+val.toString(), true); // true for asynchronous
    xmlHttp.send(null);
}


function manage_record_button()
{
    var playButton = document.getElementById('record_control');
    var record = false;
    playButton.addEventListener('click', e => {
      e.preventDefault();
      record = !record;
      if(record == true)
      {
            var xmlHttp = new XMLHttpRequest();
            xmlHttp.onreadystatechange = function()
            {
                if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
                {
                    var text = xmlHttp.responseText;
                    console.log(text);

                }
            }

            xmlHttp.open("GET", '/audio_device_start_recording', true); // true for asynchronous
            xmlHttp.send(null);
      }
      else
      {
            var xmlHttp = new XMLHttpRequest();
            xmlHttp.onreadystatechange = function()
            {
                if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
                {
                    var text = xmlHttp.responseText;
                    console.log(text);

                }
            }

            xmlHttp.open("GET", '/audio_device_stop_recording', true); // true for asynchronous
            xmlHttp.send(null);
      }
      playButton.classList.toggle('is--playing');
    });
}


function manage_live_spectrum()
{
    var channel = document.getElementById("audio_spectrum");
    var channel_height = channel.clientHeight;
    var channel_width = channel.clientWidth;
    console.log(channel_width);
    console.log(channel_height);

    var socket = io();

     socket.addEventListener('message', function (event)
     {
        console.log('Message from server ', event.data);
    });

    socket.on('connect', function()
    {
        socket.emit('get_images');
    });

    socket.on('image_upload', function(data)
    {
        console.log(data);
        var image = document.createElement('img');
        image.style.display = 'block';
        image.src = data['src'];
        image.style.width = channel_width + 'px';
        channel.insertBefore(image, channel.firstChild);
    });

}

manage_live_spectrum();
manage_record_button();
get_audio_devices();