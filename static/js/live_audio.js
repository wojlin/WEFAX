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

get_audio_devices();