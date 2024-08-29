import json

j = 0
q = []
with open("Data/mytrain.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    for i in data:
        i['index'] = j
        j += 1
        print(i)
        q.append(i)
with open("Data/mytrain.json", 'w', encoding='utf-8') as f:
    json.dump(q, f, indent=3)