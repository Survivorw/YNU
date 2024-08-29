from aiohttp import web
import logging; logging.basicConfig(level=logging.INFO)

from settings import config, BASE_DIR
from routes import setup_routes

app = web.Application()
app['config'] = config
setup_routes(app)
web.run_app(app,**app['config']['app'])