import socket
host = '127.0.0.1'
port = 9999
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((host,port))
s.listen(2)
conn,addr= s.accept()
while 1:
  print("client :"+conn.recv(1024).decode("utf-8"))
  data=input("server :")
  conn.sendall(data.encode("utf-8"))
conn.close()