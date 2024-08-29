import scrapy


class ZuowenwangNetItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


class Article(scrapy.Item):
    # 标题
    title = scrapy.Field()
    # 作者
    author = scrapy.Field()
    # 时间
    time = scrapy.Field()
    # 分区
    fenqu = scrapy.Field()
    # 分类
    fenlei = scrapy.Field()

    # url
    url = scrapy.Field()

    # 作文内容

    content = scrapy.Field()