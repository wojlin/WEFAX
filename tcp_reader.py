import socket
import json
import sys


host = ''
port = int(sys.argv[1])

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

while True:
    msg_length = client.recv(256).decode('utf-8')
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length).decode('utf-8')
        print(msg)
