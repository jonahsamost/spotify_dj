import requests
from bs4 import BeautifulSoup as bs
import httpx
from enum import Enum
import time

import logging
logger = logging.getLogger('Spotify')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

class Backend:
    def __init__(self):
        super(Backend,self).__init__()
        pass

    def get_page(self, url):
        '''
        Get url using Python Requests (http 1.1) 
        '''
        r = requests.get(url)
        if r.status_code == 200:
            return bs(r.text, 'html.parser')
        return None

    def get_pagex(self, url, follow_redirects=False):
        '''
        Get url using Python HTTPX (http 2) 
        '''
        for i in range(4):
            try:
                r = httpx.get(url,follow_redirects=follow_redirects)
                if r.status_code == 200:
                    return bs(r.text, 'html.parser')
            except Exception as e:
                logger.info(f'get_pagex fail ({url}): {e}')
                time.sleep(1)

        return bs('','html.parser')

class SqlCols(Enum):
    prim='id'
    name='name'
    href='href'
    set_href='set_href'
    mdb_id='id'
    gen_id='genre_id'
    song='song'
    song_id='song_id'
    artist='artist'
    art_id='artist_id'
    rel='relations'
    subr='subr'
    mix_url='mix_url'
    spot_sid='spot_sid'
    count='count'
    offset='offset'

    dance='danceability'
    energy='energy'
    key='pitch'
    loud='loudness'
    mode='mode'
    speech='speechiness'
    acoust='acousticness'
    instru='instrumentalness'
    live='liveness'
    valen='valence'
    tempo='tempo'
