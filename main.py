from flask import Flask, render_template, request, send_from_directory, send_file
import threading
import socket
import time

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/decode', methods=['GET'])
def decode_wefax():
    return ''


if __name__ == "__main__":
    app.run(host="localhost", port=2137, debug=True)
