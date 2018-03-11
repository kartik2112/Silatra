import socket
import sys
import timeit
from PIL import Image
jpgfile = Image.open("sachin.jpg")

port = 12348

s = socket.socket()

# s.connect(('ec2-34-215-153-222.us-west-2.compute.amazonaws.com',port))
s.connect(('127.0.0.1',port))


start_time = timeit.default_timer()
op1 = "Hellllooooooooooooooooooooooooooooo1111111111111111111111112222222222222222222233333333333333334444444444444444444444444444444444444444444444444444777777777777777777777777777777777777777777777777777777777777777777777777777777aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaajjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
s.send(op1.encode('ascii'))
print "Sent:",op1

print "Received:",s.recv(1024).decode('ascii')

print timeit.default_timer() - start_time,"s"

s.send(str(sys.getsizeof(jpgfile)).encode('ascii'))
s.send(jpgfile)
print "Sent image"

print "Image was of len:",s.recv(1024).decode('ascii')

s.close()