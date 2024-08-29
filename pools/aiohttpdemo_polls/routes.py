from aiohttp import web
from PIL import Image
from io import BytesIO

import numpy as np
import base64
routes = web.RouteTableDef()
@routes.get('/',name='index')
async def index(request):
    with open('D:\作业\大二下\gailun\pools\\aiohttpdemo_polls\\templates\handwrite\index.html', 'r', encoding='utf-8') as f:
        body = f.read()
    return web.Response(body=body, headers={'content-type': 'text/html'})

@routes.get('/img.html',name='img')
async def img(request):
    print('发送')
    with open('D:\作业\大二下\gailun\pools\\aiohttpdemo_polls\\templates\handwrite\img.html', 'r',encoding='utf-8') as f:
        body = f.read()
        return web.Response(body=body, headers={'content-type': 'text/html'})

@routes.post('/img.html')
async def imgPost(request):
    data = await request.json()
    if len(data) != 0:
        image_base64 = data['fileData']
        image_base64 = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_base64))
        image = image.convert('RGB')
        image = np.array(image)[:, :, ::-1]
        result = solve.solve(image)
        data = {'result': str(result)}
        return web.json_response(data)
    data = {'result': '错误的值'}
    return web.json_response(data)


@routes.get('/HandWrite.html')
async def hw(request):
    print('发送')
    with open('D:\作业\大二下\gailun\pools\\aiohttpdemo_polls\\templates\handwrite\HandWrite.html', 'r', encoding='utf-8') as f:
        body = f.read()
    return web.Response(body=body, headers={'content-type': 'text/html'})
@routes.post('/HandWrite.html')
async def hw(request):
    data = await request.post()
    if len(data) != 0:
        f = data['image'].file
        content = f.read()
        image = Image.open(BytesIO(content))
        image = image.convert('RGB')
        image = np.array(image)[:, :, ::-1]
        result = oneOrMore.findOneOrMore(image)
        data = {'result': str(result)}
        return web.json_response(data)
    data = await request.json()
    if len(data) != 0:
        image_base64 = data['image'].split(',')[-1]
        image_base64 = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_base64))
        image = image.convert('RGB')
        image = np.array(image)[:, :, ::-1]
        solve.solve(image,value = int(data['value']))
        data = {'result': '成功接收'}
        return web.json_response(data)
    data = {'result': '错误的值'}
    return web.json_response(data)




def setup_routes(app):
    app.add_routes(routes)
    app.add_routes([web.static('/css', path='D:\作业\大二下\gailun\pools\\aiohttpdemo_polls\\templates\handwrite\css'),
                    web.static('/img', path='D:\作业\大二下\gailun\pools\\aiohttpdemo_polls\\templates\handwrite\img')])
    routes.static('/css', path='D:\作业\大二下\gailun\pools\\aiohttpdemo_polls\\templates\handwrite\css')
    web.static('/css', path='D:\作业\大二下\gailun\pools\\aiohttpdemo_polls\\templates\handwrite\css', show_index=True)