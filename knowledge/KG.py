from py2neo import Graph, Node, Relationship
import csv
 
fr= open("G:\四人关系.csv", mode="r", encoding="gbk") # 如果路径是在工程目录下直接下相对路径，如果不在就写绝对路径
lst = []
node = []
for row in csv.reader(fr):
    lst_ = []
    lst_.append(row[0])
    lst_.append(row[1])
    lst_.append(row[2])
    lst_.append(row[3])
    lst_.append(row[4])
    lst.append(lst_)
    node.append(row[0]+' '+row[1])
    node.append(row[3]+' '+row[4])
# print(lst)
graph = Graph('bolt://localhost:7687',name="neo4j",password="******")
node = set(node) # 消除重复结点
#  建立结点：
for item in node:
    shiti,label = item.split()
    cypher_ = "CREATE (:" + label + " {name:'" + shiti + "'})     "
    graph.run(cypher_)
# 建立关系 ：
for item in lst:
    cypher_ = "MATCH  (a:" + item[1] + "),(b:" + item[4] + ") WHERE a.name = '" + item[0] + "' AND b.name = '" + item[3] + "' CREATE (a)-[r:" + item[2] + "]->(b)"
    graph.run(cypher_)