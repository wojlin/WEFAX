# WIP - WORK IN PROGRESS

### description:

tool for decoding and processing wefax


<p float="left">
  <img src="static/images/screen1.png" width="30%" />
  <img src="static/images/screen2.png" width="30%" /> 
  <img src="static/images/screen3.png" width="30%" />
</p>
<p float="left">
  <img src="static/images/screen4.png" width="30%" />
  <img src="static/images/screen5.png" width="30%" />
</p>



### theory:
WEFAX (Also known as Weatherfax, HF-FAX, Radiofax, and Weather Facsimile) is a slow scan analog image transmission mode used for the transmission of weather charts and meteorological reports

WEFAX's format is a derivative of the Automatic Picture Transmission (APT) originally developed for transmission from the polar satellites of the USA.

WEFAX has 60, 90, 100, 120, 180 and 240 LPM (Lines per minute) speeds, and two IOC (Index of Cooperation) modes, IOC 576 and IOC 288. Most weather forecasts are sent in IOC 576. 

<table cellspacing="0" cellpadding="5" border="1" align="center"><tbody><tr><th> Signal
</th>
<th> Duration
</th>
<th> IOC576
</th>
<th> IOC288
</th>
<th> Remarks
</th></tr><tr><td> Start tone
</td>
<td> 5s
</td>
<td> 300&nbsp;<span class="mw-lingo-tooltip " data-hasqtip="true"><span class="mw-lingo-tooltip-abbr">Hz</span></span>
</td>
<td> 675&nbsp;<span class="mw-lingo-tooltip " data-hasqtip="true"><span class="mw-lingo-tooltip-abbr">Hz</span></span>
</td>
<td> 200&nbsp;<span class="mw-lingo-tooltip " data-hasqtip="true"><span class="mw-lingo-tooltip-abbr">Hz</span></span> for colour fax modes.
</td></tr><tr><td> Phasing signal
</td>
<td> 30s
</td>
<td>
</td>
<td>
</td>
<td> Black line interrupted by a white pulse.
</td></tr><tr><td> Image
</td>
<td> Variable
</td>
<td> 1200 lines
</td>
<td> 600 lines
</td>
<td> At 120 lpm.
</td></tr><tr><td> Stop tone
</td>
<td> 5s
</td>
<td> 450&nbsp;<span class="mw-lingo-tooltip " data-hasqtip="true"><span class="mw-lingo-tooltip-abbr">Hz</span></span>
</td>
<td> 450&nbsp;<span class="mw-lingo-tooltip " data-hasqtip="true"><span class="mw-lingo-tooltip-abbr">Hz</span></span>
</td>
<td>
</td></tr><tr><td> Black
</td>
<td> 10s
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr></tbody></table>

### installation:
```
git clone https://github.com/wojlin/WEFAX.git
cd WEFAX
pip install flask
pip install flask_socketio
pip install numpy
pip install scipy
pip install matplotlib
pip install Pillow
pip install pyaudio
pip install wave
```

### usage:
```
python3 main.py
```


### TODO:
- f̶i̶x̶ ̶U̶I̶ ̶b̶u̶g̶ ̶w̶h̶e̶n̶ ̶f̶i̶l̶e̶n̶a̶m̶e̶ ̶i̶s̶ ̶t̶o̶o̶ ̶l̶o̶n̶g̶
- f̶i̶x̶ ̶U̶I̶ ̶b̶u̶g̶ ̶w̶h̶e̶n̶ ̶i̶m̶a̶g̶e̶ ̶f̶r̶a̶m̶e̶ ̶h̶e̶i̶g̶h̶t̶ ̶i̶s̶ ̶t̶o̶o̶ ̶b̶i̶g̶ ̶i̶n̶ ̶i̶m̶a̶g̶e̶ ̶p̶r̶e̶v̶i̶e̶w̶
- f̶i̶x̶ ̶U̶I̶ ̶b̶u̶g̶ ̶w̶h̶e̶n̶ ̶i̶m̶a̶g̶e̶ ̶f̶r̶a̶m̶e̶ ̶h̶e̶i̶g̶h̶t̶ ̶i̶s̶ ̶t̶o̶o̶ ̶b̶i̶g̶ ̶i̶n̶ ̶g̶a̶l̶l̶e̶r̶y̶
- fix convert bug when wav files have low spikes (cut unused frequencies before hilbert transform)
- fix memory allocation problem
- more responsive progress bar
- fix demodulation problem in live decoding 
- finish live decoding
- implement more test cases
- add alerts and bugs report


### BUGS AND PROBLEMS:

Port audio error in sunddevice package
```commandline
raise OSError('PortAudio library not found')
OSError: PortAudio library not found
```
solution:
```commandline
sudo apt-get install libportaudio2
sudo apt-get install libasound-dev
```