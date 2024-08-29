import requests
from bs4 import BeautifulSoup
import json

article_info = {}
finaldata = json.loads(json.dumps(article_info))
url = 'https://zuowen.xuexiaodaquan.com/e/2016/0401/7447.html'
# 获取网页数据
for index in range(100000):
    html = requests.get(url)
    html.encoding = 'utf-8'
    htmlCode = html.text
    soup = BeautifulSoup(htmlCode, 'html.parser')
    textlist = soup.find('div', class_='doc-body').find_all('p')
    texttitle = soup.find('div', class_="doc-title").text
    textlabel = soup.find('div', class_="doc-info-l").find_all('span')
    stlabel = []
    for i in textlabel:
        stlabel.append(str(i.text))
    nextone = soup.find('div', class_='doc-body').find('p', class_='next').find('a')['href']

    nextone = 'https:' + nextone
    st = ""
    for i in textlist:
        st += str(i.text)
    st.replace('<p>', '')
    st.replace("</p>", '')
    data = json.loads(json.dumps(article_info))
    data['index'] = index
    data['label'] = "生成一篇有关" + texttitle + "的" + stlabel[0] + "水平的" + stlabel[1] + stlabel[2]
    data['text'] = st
    url = nextone
    print(nextone)
    with open('Data/mytrain.json', 'a') as f:
        json.dump(data, f, indent=3)
        f.write(',\n')
