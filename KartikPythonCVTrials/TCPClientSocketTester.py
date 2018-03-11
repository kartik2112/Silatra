import socket
import sys
import timeit
# from PIL import Image
# jpgfile = Image.open("sachin.jpg")

myfile = open("sachin.jpg", 'rb')
bytes123 = myfile.read()

port = 12349

s = socket.socket()

s.connect(('ec2-54-187-180-129.us-west-2.compute.amazonaws.com',port))
# s.connect(('127.0.0.1',port))


start_time = timeit.default_timer()
op1 = "Hellllo"
s.send(op1.encode('ascii'))
print "Sent:",op1

print "Received:",s.recv(1024).decode('ascii')

print "Sending image of size:",len(bytes123)

s.send(str(len(bytes123)).encode('ascii'))
s.sendall(bytes123)

print "Sent image"

print "Received ACK that image was of len:",s.recv(1024).decode('ascii')

print "Took:",timeit.default_timer() - start_time,"s"

s.close()