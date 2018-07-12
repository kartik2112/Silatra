from flask import Flask, render_template
from random import randint
import os
import socket
from contextlib import closing

dir_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)

def check_socket(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('', port))
        sock.close()
        return True
    except:
        return False



silatra_app_path = dir_path+'/Receiver.py'
pythonExecPath = '/home/kartik/.virtualenvs/cv/bin/python3'

# @app.route('/')
# def index():
#   return render_template('template.html')

portsInUseBySilatra = {}

@app.route('/get-port-number/')
def my_link():
    while(True):
        portNo = randint(40000,50000)
        if portNo not in portsInUseBySilatra:
            if(check_socket(portNo)):
                os.system('nohup '+pythonExecPath+' '+silatra_app_path+' --portNo '+str(portNo)+' --displayWindows False &')
                portsInUseBySilatra[portNo] = '1'
                break
    
    return str(portNo)

# @app.route('/verify-silatra-server/')
# def my_link():
#     return "Connected to Silatra Server"

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host = '0.0.0.0')