import urllib.request
from bs4 import BeautifulSoup
import os
import json
url="https://www.zuowenwang.net/"
head = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.42"
        }
request=urllib.request.Request(url,headers=head)
html=""
response=urllib.request.urlopen(request)
html=response.read().decode("utf-8")
bs=BeautifulSoup(html,"html.parser")

text=bs.find_all('p')[:-1]

content=''
for p in text:
    content+=p.text.strip()
content="\"content\":"+content

os.makedirs('文件', exist_ok=True)
with open(f"文件/result.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(content,indent=2,ensure_ascii=False))
print(content)