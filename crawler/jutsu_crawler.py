import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    name = 'narutospider'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu'] ## Initial webpage to start scraping from. 

    def parse(self, response):

        ## Iterate over page list items store these values in  a dicitonary. 
        for href in response.css('.smw-columnlist-container')[0].css('a::attr(href)').extract():

            extracted_data = scrapy.Request('https://naruto.fandom.com' + href, callback=self.parse_jutsu)

            yield extracted_data

        # Iterate over anchor tags to select next page. 
        for next_page in response.css('a.mw-nextlink'): ## specified anchor tag from webpage. 
            yield response.follow(next_page, self.parse)

    
    def parse_jutsu(self, response):

        jutsu_name = response.css('span.mw-page-title-main::text').extract()[0]
        jutsu_name = jutsu_name.strip()

        div_selector = response.css('div.mw-parser-output')[0]
        div_html = div_selector.extract()

        soup = BeautifulSoup(div_html).find('div')
        
        jutsu_type = ''

        if soup.find('aside'):

            aside = soup.find('aside')

            for cell in aside.find_all('div', {'class' : 'pi-data'}):

                if cell.find('h3'):

                    cell_name = cell.find('h3').text.strip()

                    if cell_name == 'Classification':

                        jutsu_type = cell.find('div').text.strip()

        soup.find('aside').decompose()

        justsu_desc = soup.text.strip()
        justsu_desc.split('Trivia')[0].strip()

        return dict(
            jutsu_name = jutsu_name, 
            jutsu_type = jutsu_type, 
            justsu_desc = justsu_desc
        )
