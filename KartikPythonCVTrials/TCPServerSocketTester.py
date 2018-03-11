import socket
from PIL import Image
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
op1 = "HELLO_Hellllooooooooooooooooooooooooooooo1111111111111111111111112222222222222222222233333333333333334444444444444444444444444444444444444444444444444444777777777777777777777777777777777777777777777777777777777777777777777777777777aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaajjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
client.send(op1.encode('ascii'))
print "Sent:",op1

# buf = client.recv(4)

    
# print(buf)
size = int(client.recv(1024).decode('ascii'))
data = client.recv(size,socket.MSG_WAITALL)

client.send(str(sys.getsizeof(data)).encode('ascii'))

s.close()