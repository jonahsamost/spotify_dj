import urllib , logging, inspect
from bs4 import BeautifulSoup as bs
from urllib import request as rq


def get_page(url):
    r = rq.urlopen(url)
    if (r.code != 200):
        logging.info("{}: return code not 200 for url: {}".format(inspect.stack()[0][3] , url))
        return None
    logging.info("{}: return code 200 for url: {}".format(inspect.stack()[0][3] , url))
    return r.read()

'''
returns BS object encapsulating html
'''
def get_html(page):
    return bs(page,'html.parser')

def get_bs_class(soup,level,classname):
    return soup.findAll(level, {'class' : classname })

def get_bs_tag(soup,tag):
    return soup.findAll(tag)

def create_song(song,artist,year):
    return {'song':song,'artist':artist,'year':year}

def search_year(txt):
    year = -1
    if txt:
        pos = 0 
        while 1:
            pre,post = txt.find('(',pos), txt.find(')',pos)
            if pre == -1 or post == -1: break 
            if post - pre == 5:
                try:
                    year = int(txt[pre + 1 : post])
                    break 
                except:
                    pass
            pos = pre + 1 
    return year

def get_year_from_item(item):
    try:
        return int(item['album']['release_date'].split('-')[0])
    except:
        return -1

