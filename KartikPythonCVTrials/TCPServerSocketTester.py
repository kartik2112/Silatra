import socket
# from PIL import Image
import sys


port = 12348


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         
print "TCP Socket successfully created"
s.bind(('', port))        
print "TCP Socket binded to %s" %(port)
s.listen(1)     
print "Socket is listening"
client, addr = s.accept()     
print 'Got TCP connection from', addr

print "Received:",client.recv(1024).decode('ascii')
op1 = "HELLO"
client.send(op1.encode('ascii'))
print "Sent:",op1


    
# print(buf)
size = int(client.recv(1024).decode('ascii'))
print "Receiving image of size:",size

data = client.recv(size,socket.MSG_WAITALL)

print "Received image"

client.send(str(len(data)).encode('ascii'))

print "ACKing image received of size",len(data)

s.close()