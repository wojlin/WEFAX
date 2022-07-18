import datetime
import threading
import os
from flask import Flask, send_from_directory, render_template, request, send_file
from werkzeug.utils import secure_filename
import shutil
import time
import pyaudio
from flask_socketio import SocketIO

from flask_socketio import emit
import ast
from distutils.dir_util import copy_tree
from zipfile import ZipFile
import zipfile
import logging
import sys
import signal

from wefax import Demodulator
from wefax_live import LiveDemodulator


def stop(signum, frame):
    print('exit')
    sys.exit(0)


logging.getLogger('werkzeug').disabled = True
signal.signal(signal.SIGINT, stop)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

demodulators = {}


def delete_directory(path: str, datestamp: str):
    print(f"{path} deleted")
    del demodulators[datestamp]
    shutil.rmtree(path)


@socketio.on('get_progress')
def get_progress(json):
    try:
        print('received json: ' + str(json))
        demodulator = demodulators[json['data']]
        print(demodulator.file_info())
        notConverted = True
        while notConverted:
            if len(demodulator.websocket_stack) > 0:
                emit('upload_progress', demodulator.websocket_stack[0])
                print(demodulator.websocket_stack[0])
                if demodulator.websocket_stack[0]['data_type'] == 'message':
                    print("MESSAGE")
                    if demodulator.websocket_stack[0]['message_content'] == 'convert_end':
                        print("END")
                        threading.Timer(3600, delete_directory,
                                        args=(f"static/temp/{json['data']}", json['data'],)).start()
                        notConverted = False
                demodulator.websocket_stack.pop(0)
    except Exception as e:
        print(e)


@socketio.on('get_spectrum')
def get_images():
    global live_demodulator
    try:
        notConverted = True
        while notConverted:
            if len(live_demodulator.spectrum_websocket_stack) > 0:
                emit('spectrum_upload', live_demodulator.spectrum_websocket_stack[0])
                live_demodulator.spectrum_websocket_stack.pop(0)
    except Exception as e:
        print(e)


@socketio.on('get_frames')
def get_frames():
    global live_demodulator
    try:
        notConverted = True
        while notConverted:

            if len(live_demodulator.frames_websocket_stack) > 0:
                print('frame upload', live_demodulator.frames_websocket_stack[0])
                emit('frame_upload', live_demodulator.frames_websocket_stack[0])
                live_demodulator.frames_websocket_stack.pop(0)
    except Exception as e:
        print(e)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/images'),
                               'favicon.png', mimetype='image/vnd.microsoft.icon')


@app.route('/delete_entry', methods=['POST'])
def delete_entry():
    r = request.form
    entry = r['datestamp']
    delete_path = f'static/gallery/{entry}'
    shutil.rmtree(delete_path)
    return "success"


@app.route('/save_to_gallery', methods=['POST'])
def save_to_gallery():
    r = request.form
    datestamp = r['datestamp']
    folder_path = f'static/temp/{datestamp}/'
    out_path = f'static/gallery/{datestamp}/'
    copy_tree(folder_path, out_path)
    file_info = demodulators[datestamp].file_info()
    file_info['filename'] = str(file_info['filename']).split('.')[0]
    audio_filename = str(file_info['filename']) + '.wav'
    image_filename = str(file_info['filename']) + '.png'
    file_info['audio_filename'] = audio_filename
    file_info['image_filename'] = image_filename
    file_info['date'] = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    with open(out_path + 'info.json', 'w') as f:
        f.write(str(file_info))
        f.close()

    with ZipFile(f'{out_path}/files.zip', 'w') as zipObj:
        zipObj.write(os.path.join(out_path, audio_filename), audio_filename, zipfile.ZIP_DEFLATED)
        zipObj.write(os.path.join(out_path, image_filename), image_filename, zipfile.ZIP_DEFLATED)
        zipObj.write(os.path.join(out_path, 'info.json'), 'info.json', zipfile.ZIP_DEFLATED)
        zipObj.close()

    return f'files saved to {out_path}'


@app.route('/get_gallery_files', methods=['GET'])
def get_gallery_files():
    dirnames = [str(f.path).split('/')[-1] for f in os.scandir('static/gallery') if f.is_dir()]
    gallery_files = {}
    for directory in dirnames:
        with open(f'static/gallery/{directory}/info.json', 'r') as f:
            ret_json = ast.literal_eval(f.read())
        gallery_files[directory] = ret_json
    gallery_files = str(dict(sorted(gallery_files.items(), key=lambda x: x[0].lower(), reverse=True))).replace("'", '"')
    print(gallery_files)
    return gallery_files


@app.route('/file_converter')
def file_converter():
    return render_template('file_converter.html')


@app.route('/get_audio_devices')
def get_audio_devices():
    p = pyaudio.PyAudio()
    return {i: p.get_device_info_by_index(i) for i in range(p.get_device_count())}


@app.route('/audio_device_start_recording')
def audio_device_start_recording():
    global live_demodulator
    return live_demodulator.start_recording()


@app.route('/audio_device_stop_recording')
def audio_device_stop_recording():
    global live_demodulator
    return live_demodulator.stop_recording()


@app.route('/get_combined_audio_file')
def get_combined_audio_file():
    global live_demodulator
    return live_demodulator.combine()


@app.route('/get_audio_info')
def get_audio_info():
    global live_demodulator
    return live_demodulator.audio_info()


@app.route('/change_audio_device/<device_index>')
def change_audio_device(device_index):
    global live_demodulator
    return live_demodulator.connect(int(device_index))


@app.route('/live_converter')
def live_converter():
    global live_demodulator
    if live_demodulator is not None:
        live_demodulator.stop_recording()
        live_demodulator.end_stream()
    live_demodulator = LiveDemodulator(path=save_directory, tcp_stream=True)
    return render_template('live_converter.html')



@app.route('/gallery')
def gallery():
    return render_template('gallery.html')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/convert_file', methods=['POST'])
def convert_file():
    try:
        r = request.form
        datestamp = r['datestamp']
        demodulator = demodulators[datestamp]

        print('############')
        print('converting')
        print(demodulator.file_info())
        print('############')
        demodulator.process()
        filename = str(demodulator.file_info()['filename']).split('.')[0]
        output_filepath = f'static/temp/{datestamp}/{filename}.png'
        demodulator.save_output_image(output_filepath)
        filename = str(demodulator.file_info()['filename']).split('.')[0] + '.png'

        return {"output_src": output_filepath, "output_name": filename}
    except Exception as e:
        print(e)
        return e


@app.route('/load_file', methods=['POST'])
def load_file():
    try:
        r = request.form
        dir_name = str(round(time.time() * 1000))
        save_directory = f"static/temp/{dir_name}"
        os.mkdir(save_directory)

        path = os.path.join(save_directory, secure_filename(request.files['file'].filename))
        request.files['file'].save(path)

        demodulator = Demodulator(path, lines_per_minute=120, tcp_stream=True, quiet=True)

        ret = demodulator.file_info()
        ret['datestamp'] = dir_name
        demodulators[dir_name] = demodulator

        print('############')
        print(f"filename: {secure_filename(request.files['file'].filename)}")
        print('############')
        return ret
    except Exception as e:
        print(e)
        return e


if __name__ == "__main__":
    try:
        shutil.rmtree('static/temp/')
    except Exception:
        pass
    os.mkdir('static/temp')

    dir_name = str(round(time.time() * 1000))
    save_directory = f"static/temp/{dir_name}/"
    os.mkdir(save_directory)

    live_demodulator = LiveDemodulator(path=save_directory, tcp_stream=True)

    socketio.run(app, host="localhost", port=2137, debug=True)
