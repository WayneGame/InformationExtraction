import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule
"""
class IeSpider(scrapy.Spider):
    name = "ie"

    start_urls=[
        "http://stadt-verzeichnis.de/"
    ]


    def parse(self, response):
        for citys in response.css('table li')[:26]:
            next_site = citys.css('a::attr(href)').get('').strip()
            if next_site is not None and next_site != 'http://www.seo-factory.de' and next_site != 'http://postleitzahl-plz.verzeichniss.net' and next_site != 'http://www.lasershow24.de/robots.txt':
                try:
                    yield response.follow(url=next_site, callback=self.parse_city)
                except:
                    pass


    def parse_city(self, response):
        for city in response.css('table[width="570"] tr td a').getall():
            city_site = city.split('\"')[1]
            if city_site is not None and city_site != 'http://www.seo-factory.de' and city_site != 'http://postleitzahl-plz.verzeichniss.net' and city_site != 'http://www.lasershow24.de/robots.txt':
                try:
                    yield response.follow(url=city_site, callback=self.city_page)
                except:
                    pass


    def city_page(self, response):
        kreisfrei = response.css('font[size="2"]::text').getall().count('freie Stadt \r\n')
        if kreisfrei:
            city_name = response.css('table[width="650"] tr td h1::text').get()[6:].replace(" ", "")
            link = response.xpath("//*[contains(text(), 'Stadt-Homepage')]").get().split('\"')[1]
            yield {
                'type': "City",
                'link': link,
                'name': city_name
            }
"""
"""
class IeSpider(scrapy.Spider):
    name = "ie"

    start_urls=[
        "http://www.landraete.de/"
    ]


    def parse(self, response):
        for citys in response.css('table tr'):
            link = citys.css('td a::text').get()
            city_name = citys.css('td::text').get()
            if link != 'Zum Seitenanfang' and link is not None:
                yield {
                    'type': "Kreis",
                    'link': link,
                    'name': city_name
                }
"""

class IeSpider(scrapy.Spider):
    name = "ie"

    allowed_domains = ['books.toscrape.com']
    start_urls = ['http://books.toscrape.com/']
    base_url = 'http://books.toscrape.com/'
    rules = [Rule(LinkExtractor(allow='catalogue/'),
                  callback='parse_item', follow=True)]

    def parse_item(self, response):
        print("URL: " + response.url)
        with open("test.html", 'wb') as f:
            f.write(response.url)