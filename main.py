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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/decode', methods=['POST'])
def decode():
    r = request.form
    print(request.files)
    dir_name = str(round(time.time() * 1000))
    save_directory = f"temp/{dir_name}"
    os.mkdir(save_directory)
    print(secure_filename(request.files['file'].filename))
    path = os.path.join(save_directory, secure_filename(dir_name))
    request.files['file'].save(path)

    demodulator = Demodulator('input/input_lq.wav',
                              lines_per_minute=120,
                              tcp_stream=False,
                              tcp_host='localhost',
                              tcp_port=2000)

    return demodulator.file_info()

if __name__ == "__main__":
    app.run(host="localhost", port=2137, debug=True)
