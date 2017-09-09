#coding:utf-8
from socket import *
from time import ctime
import struct

HOST = '192.168.0.102'  # 主机
PORT = 5099 # 端口号，可以随意选择
BUFSIZ = 1024
ADDR = (HOST, PORT) #主机端口号组成一个套接字地址

tcpSerSock = socket(AF_INET, SOCK_STREAM) #创建一个套接字对象，是AF_INET族的tcp套接字
tcpSerSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
tcpSerSock.bind(ADDR) #这个函数用于绑定地址到套接字
tcpSerSock.listen(5) # 服务器开始监听连接，参数表示最多允许同时有几个连接进来

while True:
    print ('waiting for connection...')
    tcpCliSock, addr = tcpSerSock.accept() #用于等待连接的到来
    print ('...connected from:',addr)

    while True:
        data = tcpCliSock.recv(BUFSIZ)
        data1 = struct.unpack("4d", data)[0]
        data2 = struct.unpack("4d", data)[1]
        data3 = struct.unpack("4d", data)[2]
        
        tcpCliSock.send(str(10).encode())  # 将时间戳作为内容发送给客户端
         #用于接收从客户端发来的数据 参数代表一次最多接受的数据量，这里为1k
        print(type(data1),data1)
        print(type(data2),data2)
        print(type(data3),data3)
        if not data:
            break
        #print(data)
        # tcpCliSock.send('[%s] %s' % (ctime().encode(), data.encode())) # 将时间戳作为内容发送给客户端
        # tcpCliSock.send(ctime().encode())  # 将时间戳作为内容发送给客户端

    tcpCliSock.close()

tcpSerSock.close()