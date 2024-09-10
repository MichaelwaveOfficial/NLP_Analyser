from bs4 import BeautifulSoup 

class DataCleaner(object):

    
    def __init__(self) -> None:
        pass 


    def insert_line_breaks(self, text):
        return text.replace('<\p>', '<\p>\n')
    

    def remove_html_tags(self, text):
        return BeautifulSoup(text, 'lxml').text


    def clean_text(self, text):
        text = self.insert_line_breaks(text=text)
        text = self.remove_html_tags(text=text)
        text.strip()

        return text 
    