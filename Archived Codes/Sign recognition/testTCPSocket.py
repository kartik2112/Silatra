import socket
import struct
import sys
import netifaces as ni
import os
import tkinter



# Comment these 3 stmts if you dont have numpy, cv2, imutils
import numpy as np
import imutils
import cv2


port = 49164



def port_initializer():
    global port
    port = int(port_entry.get())
    opening_window.destroy()


opening_window = tkinter.Tk()
port_label = tkinter.Label(opening_window, text = "Port to be reserved:")
port_label.pack(side = tkinter.LEFT)
port_entry = tkinter.Entry(opening_window, bd=3)
port_entry.pack(side = tkinter.RIGHT)
save_button = tkinter.Button(opening_window, command = port_initializer)
save_button.pack()
opening_window.mainloop()
ni.ifaddresses('wlo1')
ipAddr = ni.ifaddresses('wlo1')[ni.AF_INET][0]['addr']


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         
print("TCP Socket successfully created")
s.bind(('', port))        
print("TCP Socket binded to %s: %s" %(ipAddr,port))
s.listen(1)     
print("Socket is listening")
client, addr = s.accept()     
print('Got TCP connection from', addr)


while True:

    buf = client.recv(4)
    print(buf)
    size = struct.unpack('!i', buf)[0]  
    print("receiving image of size: %s bytes" % size)


    if(size == 0 ):
        op1 = "QUIT\r\n"
        client.send(op1.encode('ascii'))
        break

    data = client.recv(size,socket.MSG_WAITALL)
    op1 = str(size)+" bytes\r\n"
    client.send(op1.encode('ascii'))



    # Comment these 5 stmts if you dont have numpy, cv2, imutils
    nparr = np.fromstring(data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_np = imutils.rotate_bound(img_np,90)
    img_np = cv2.resize(img_np,(0,0), fx=0.7, fy=0.7)
    cv2.imshow("Img",img_np)

    k = cv2.waitKey(10)
    if k == 'q':
        break


print('Stopped TCP server of port: '+str(port))