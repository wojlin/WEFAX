var animate_0_25_b = document.getElementById("animate-0-25-b");
var animate_25_50_b = document.getElementById("animate-25-50-b");
var animate_50_75_b = document.getElementById("animate-50-75-b");
var animate_75_100_b = document.getElementById("animate-75-100-b");
var percentage = document.getElementById("percentage_text");

function renderProgress(progress)
{
    progress = Math.floor(progress);
    if(progress<25){
        var angle = -90 + (progress/100)*360;
        animate_0_25_b.style.transform = "rotate("+angle+"deg)";
    }
    else if(progress>=25 && progress<50){
        var angle = -90 + ((progress-25)/100)*360;
         animate_0_25_b.style.transform = "rotate(0deg)";
         animate_25_50_b.style.transform = "rotate("+angle+"deg)";
    }
    else if(progress>=50 && progress<75){
        var angle = -90 + ((progress-50)/100)*360;
        animate_0_25_b.style.transform = "rotate(0deg)";
        animate_25_50_b.style.transform = "rotate(0deg)";
        animate_50_75_b.style.transform = "rotate("+angle+"deg)";
    }
    else if(progress>=75 && progress<=100){
        var angle = -90 + ((progress-75)/100)*360;
        animate_0_25_b.style.transform = "rotate(0deg)";
        animate_25_50_b.style.transform = "rotate(0deg)";
        animate_50_75_b.style.transform = "rotate(0deg)";
        animate_75_100_b.style.transform = "rotate("+angle+"deg)";
    }
    percentage.innerHTML = progress+"%";
}