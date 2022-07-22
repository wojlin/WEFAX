var animate_0_25_b = document.getElementById("animate-0-25-b");
var animate_25_50_b = document.getElementById("animate-25-50-b");
var animate_50_75_b = document.getElementById("animate-50-75-b");
var animate_75_100_b = document.getElementById("animate-75-100-b");

function renderProgress(progress)
{
    progress = Math.floor(progress);

    let progressBar = document.getElementById("circular-progress");
    let valueContainer = document.getElementById("value-container");


    progressBar.style.background = `conic-gradient(
          rgba(88, 88, 88, 1) ${progress * 3.6}deg,
          rgba(18, 18, 18, 1) ${progress * 3.6}deg
      )`;

    valueContainer.innerHTML = progress+"%";
}