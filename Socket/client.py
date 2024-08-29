import socket
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect(('127.0.0.1', 9999))
while 1:
  data=input("client : ")
  c.send(data.encode("utf-8"))
  result=c.recv(1024)
  print("server : "+result.decode("utf-8"))
c.close