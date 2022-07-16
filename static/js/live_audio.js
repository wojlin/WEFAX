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

    var freq_min = document.getElementById('freq_min');
    var freq_max = document.getElementById('freq_max');
    var time_min = document.getElementById('time_min');
    var time_max = document.getElementById('time_max');

    var spectrum_window = document.getElementById("spectrum_window");
    var window_height = spectrum_window.clientHeight;

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
         var channel_height = channel.clientHeight;
    var channel_width = channel.clientWidth;
        var segment_duration = data['length'];
        var segment_height = data['height'];
        var time_max_var = '-' + (window_height  * segment_duration / segment_height).toFixed(1).toString() + 's';
        time_max.innerHTML = time_max_var;
        image.style.display = 'block';
        image.src = data['src'];
        image.style.width = channel_width + 'px';
        channel.insertBefore(image, channel.firstChild);
        spectrum_bottom = parseInt(channel.style.bottom, 10);
        console.log(window_height)
        console.log(channel_height)
        channel.style.bottom = -channel_height + window_height + 'px';

    });

}

manage_live_spectrum();
manage_record_button();
get_audio_devices();