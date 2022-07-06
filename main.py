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
from flask_socketio import SocketIO
from wefax import Demodulator
from flask_socketio import send, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

demodulators = {}

##
@socketio.on('get_progress')
def handle_my_custom_event(json):
    print('received json: ' + str(json))
    demodulator = demodulators[json['data']]
    print(demodulator.file_info())
    while True:
        if len(demodulator.websocket_stack) > 0:
            emit('upload_progress', demodulator.websocket_stack[0])
            demodulator.websocket_stack.pop(0)


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
    print(demodulator.file_info())
    print('############')
    demodulator.process()
    output_filepath = f'temp/{datestamp}/output.png'
    demodulator.save_output_image(output_filepath)
    filename = str(demodulator.file_info()['filename']).split('.')[0] + '.png'
    return {"output_src": output_filepath, "output_name": filename}


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
    demodulators[dir_name] = demodulator

    print('############')
    print(f"filename: {secure_filename(request.files['file'].filename)}")
    print('############')
    return ret


if __name__ == "__main__":
    socketio.run(app, host="localhost", port=2137, debug=True)
