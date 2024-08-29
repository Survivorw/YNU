import re
import urllib.request
import requests
import json
from bs4 import BeautifulSoup
import jieba

import matplotlib.pyplot as plt
import os
baseurl="https://www.zuowenwang.net/"
lid_dic = {'中学'}
lid=lid_dic["中学"]
def get_url(lid,page,pageid=153): # 每一页的pageid始终为153  
    
    return baseurl +"&lid="+str(lid)+"&pageid="+str(pageid)+"&page="+str(page)

def get_json_url(url):
        
    Art_data_list = []
    json_req = requests.get(url)
    user_json = json.loads(json_req.text)
    for data in user_json["result"]["data"]:
        Art_data_list.append([data["url"],data['keywords'],data['title']])
        with open(f"文件/keywords.txt",'a',encoding="utf-8") as f:
            f.write(data['keywords'])
    return Art_data_list

def askURL(url):
    
    head = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.42"
        }
    request=urllib.request.Request(url,headers=head)
    html=""
    response=urllib.request.urlopen(request)
    html=response.read().decode("utf-8")
    
    return html

def getData(Art_data_list):

    for Art_data in Art_data_list:
        html=askURL(Art_data[0])
        bs=BeautifulSoup(html,"html.parser")    
        
        title="\"title\":\""+Art_data[2]+"\""#title获取
        
        url="\"url\":\""+Art_data[0]+"\""
            
        topic="\"topic\":"+"\"\""
        try:
            text=bs.find_all('p',attrs={'cms-style':'font-L'} )
            
            content=''
            for p in text:
                content+=p.text.strip()
            content="\"content\":\""+content+"\""
                        #content获取
        except:
            pass
        json_data="{"+title+",\n"+content+",\n"+topic+",\n"+url+"}"
        if len(content)<50: continue 
        try:
            json_data = json.loads(json_data)
            os.makedirs('文件', exist_ok=True)
            with open(f"文件/result.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(json_data,indent=2,ensure_ascii=False))
                f.write(",\n")
        except:
            pass
        


def main():
    with open(f"文件/result.json", "a", encoding="utf-8") as f:
        f.write("[  \n")
    for page in range(21,61):  
        stock_url = get_url(lid,page)
        
        Art_data_list = get_json_url(stock_url)
        
        getData(Art_data_list)
    with open(f"文件/result.json", "a", encoding="utf-8") as f:       
        f.write("]")
    
    generate_wc()        

if __name__=="__main__":
     main()