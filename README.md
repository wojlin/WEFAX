# WIP - WORK IN PROGRESS

### description:

tool for decoding and processing wefax

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
pip install -r requirements.txt
```

### usage:
```
python3 main.py
```

### TODO:
- more responsive progress bar
- live decoding 
- tweaks in frontend style
- create tests
- add alerts and bugs report