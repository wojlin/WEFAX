from flask import Flask, render_template, request, send_from_directory, send_file
import threading
import socket
import time
import os
from flask import Flask, send_from_directory, render_template, request, send_file
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import shutil
import time

from wefax import Demodulator

app = Flask(__name__)

demodulators = {}


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/images'),
                               'favicon.png', mimetype='image/vnd.microsoft.icon')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/convert_file', methods=['POST'])
def convert_file():

    r = request.form
    datestamp = r['datestamp']
    demodulator = demodulators[datestamp]

    print('############')
    print('converting')
    print(demodulator.get_tcp_port())
    print(demodulator.file_info())
    print('############')
    demodulator.process()
    output_filepath = f'temp/{datestamp}/output.png'
    demodulator.save_output_image()
    return send_from_directory('', output_filepath)


@app.route('/load_file', methods=['POST'])
def load_file():

    r = request.form
    dir_name = str(round(time.time() * 1000))
    save_directory = f"temp/{dir_name}"
    os.mkdir(save_directory)

    path = os.path.join(save_directory, secure_filename(request.files['file'].filename))
    request.files['file'].save(path)

    demodulator = Demodulator(path, lines_per_minute=120, tcp_stream=True)

    ret = demodulator.file_info()
    ret['datestamp'] = dir_name
    ret['tcp_port'] = demodulator.get_tcp_port()
    demodulators[dir_name] = demodulator

    print('############')
    print(f"filename: {secure_filename(request.files['file'].filename)}")
    print(f'tcp port: {ret["tcp_port"]}')
    print('############')
    return ret


if __name__ == "__main__":
    app.run(host="localhost", port=2137, debug=True)
