# Reference: https://stackoverflow.com/a/23312964/5370202

import socket
import struct

import numpy as np
import cv2
import imutils


preds = []
maxQueueSize = 15
noOfSigns = 128
minModality = int(maxQueueSize/2)

UDP_IP_ADDRESS = ""  #IP Address of the Server, i.e. The Laptop
UDP_IP_ADDRESS2 = "192.168.2.6" #IP Address of the Client, i.e. The Smartphone
UDP_RECEIVE_PORT_NO = 9005          #Port for Image transmission (Smartphone -> Laptop)
UDP_SEND_PORT_NO = UDP_RECEIVE_PORT_NO #Port for Text Transmission (Laptop -> Smartphone)


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)         
print("Socket successfully created")

s.bind((UDP_IP_ADDRESS,  UDP_RECEIVE_PORT_NO))        
print("socket binded to %s" %(UDP_RECEIVE_PORT_NO))
 
while True:
    

    #if(size == 0):
     #   op1 = "QUIT\r\n"
      #  client.send(op1.encode('ascii'))
       # break

    
    print("Receiving data")
    data, addr = s.recvfrom(65507)
    print("Receiving data2")

    # Instead of storing this image as mentioned in the 1st reference: https://stackoverflow.com/a/23312964/5370202
    # we can directly convert it to Opencv Mat format
    #Reference: https://stackoverflow.com/a/17170855/5370202
    nparr = np.fromstring(data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_np = imutils.rotate_bound(img_np,90)

    cv2.imshow("Img",img_np)
    
    Message = bytearray([1,2,3,4,5])
    clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    clientSock.sendto(Message, (str(addr), UDP_SEND_PORT_NO))
    print("Sending data")

    # client.send(pred.encode('ascii'))



    if cv2.waitKey(10) == 'q':
        break
    


print('received, yay!')

# client.close()
s.close()
cv2.destroyAllWindows()

