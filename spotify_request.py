import requests
import json
import time
from threading import Lock
import pickle
import spotipy
import os
import shutil

import logging
logger = logging.getLogger('Spotify')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

from spotify_app_dic import apps


class SpotifyRequest(object):
    def __init__(self, username):
        super(SpotifyRequest, self).__init__()
        self._token = None
        self._base = 'https://api.spotify.com/v1/'
        self._session = requests.Session()
        self._lock = Lock()
        self._username = username
        self._soa = None

        self._loc = 0
        # self.reinit()

    def reauth(self):
        # if self._loc == len(apps):
        #     logger.info("!!!!! PAST ALL APPS")
        #     os._exit(1)

        dic = apps[self._loc]
        self._client_id = dic['id']
        self._secret = dic['secret']
        self._fname = dic['name']

        fname = f".{dic['name']}_cache"
        shutil.copy2(fname, '.cache')

        self._redirect = 'https://jasvandy.github.io/'

        logger.info(f"apps init {self._loc} : {dic['name']}")
        self._soa = spotipy.oauth2.SpotifyOAuth(client_id=self._client_id, 
            client_secret=self._secret, redirect_uri=self._redirect, 
            scope=self.scope)
        '''
        scope = [ 'user-read-private', 'user-follow-read', 'user-library-read', 'user-top-read', 'user-read-recently-played', 'playlist-modify-private', 'playlist-read-private', 'playlist-modify-public' ]
        scope = ' '.join(scope)
        _soa = spotipy.oauth2.SpotifyOAuth(client_id=_client_id, 
            client_secret=_secret, redirect_uri=_redirect, 
            scope=scope)
        '''

        self._loc = (1 + self._loc) % len(apps)

        return 1

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, tok):
        self._token = tok

    @property
    def scope(self):
        scope = [
            'user-read-private',
            'user-follow-read',
            'user-library-read',
            'user-top-read',
            'user-read-recently-played',
            'playlist-modify-private',
            'playlist-read-private',
            'playlist-modify-public'
        ]
        return ' '.join(scope)

    @property
    def auth(self):
        assert self._token is not None, 'self._token is None'
        return {"Authorization": "Bearer {0}".format(self.token)}

    def reinit(self):
        logger.info("REINIT!!")

        def __tokes_init():
            # while 1:
            #     try:
            #         logger.info("soa.get_cached_token")
            #         tokes = self._soa.get_cached_token()
            #         break
            #     except:
            #         logger.info("exception on soa.get_cached_token")
            #         if not self.reauth():
            #             raise Exception('REINIT FAIL')
            self.reauth()
            tokes = None
            try:
                tokes = self._soa.get_cached_token()
            except:
                pass

            if not tokes:
                self._soa.get_access_token()
                tokes = self._soa.get_cached_token()

            return tokes

        self._lock.acquire()

        tokes = __tokes_init()

        while 1:
            try:
                logger.info("Refresh access token")
                self._soa.refresh_access_token(tokes['refresh_token'])
                break
            except Exception as e:
                logger.info(f"FAIL : Refresh access token {e}")
                tokes = __tokes_init()

        tokes = self._soa.get_cached_token()
        self.token = tokes['access_token']

        self._lock.release()
    

    def _call(self, method, url, args=None, payload=None, **kwargs):
        url = self._base + url
        while 1:
            headers = self.auth
            headers['Content-Type'] = 'application/json'

            self._lock.acquire()
            try:
                response = self._session.request(method, url, headers=headers, params=kwargs)
            except Exception as e:
                logger.info(f'EXCEPTION ON REQUEST: {e}')
                self._session = requests.Session()
                time.sleep(10)
                self._lock.release()
                continue 

            self._lock.release()

            if response.status_code == 200:
                return json.loads(response.text)
            elif response.status_code == 429:
                sleep_time = response.headers['Retry-After']
                logger.info(f"!!!!!!! HIT RATE LIMIT...SLEEPING FOR {sleep_time} SECONDS")
                self.reinit()
            else:
                logger.info("Error: {response.status_code}, {response.text}")
                return None

    def current_user_followed_artists(self, limit=50, after=0):
        return self._call('GET', 'me/following', type='artist', limit=limit, after=after)

    def current_user(self):
        return self._call('GET', 'me/')

    def current_user_top_artists(self, limit=50, offset=0, time_range='short_term'):
        return self._call('GET', 'me/top/artists', limit=limit, offset=0, time_range=time_range)

    def current_user_top_tracks(self, limit=50, offset=0, time_range='short_term'):
        return self._call('GET', 'me/top/tracks', limit=limit, offset=0, time_range=time_range)

    def current_user_saved_tracks(self, limit=50, offset=0, market=None):
        return self._call('GET', 'me/tracks', limit=limit, offset=0, market=market)

    def user_playlists(self, user, limit=50, offset=0):
        return self._call('GET', f'users/{user}/playlists', limit=limit, offset=0)

    def playlist_items(self, pid, fields=None, limit=50, offset=0, market=None, additional_types='track,episode'):
        return self._call('GET', f'playlists/{pid}/tracks', fields=fields, limit=limit, offset=offset, market=market, additional_types=additional_types)

    def artists(self, artists):
        ids = ','.join(artists)
        return self._call('GET', f'artists/?ids={ids}')

    def artist_albums(self, artist_id, album_type=None, country=None, limit=50, offset=0):
        return self._call('GET', f'artists/{artist_id}/albums', 
            album_type=album_type, country=country, limit=limit, offset=offset)

    def albums(self, albums):
        ids = ','.join(albums)
        return self._call('GET', f'albums/?ids={ids}')

    def search(self, q, limit=50, offset=0, type='track', market=None):
        return self._call('GET', 'search', q=q, limit=limit, offset=offset, type=type, market=market)

    def audio_features(self, tracks=[]):
        trks = ','.join(tracks)
        return self._call('GET', f'audio-features/?ids={trks}')
    
    def album_tracks(self, album_id, limit=50, offset=0, market=None):
        return self._call('GET',f'albums/{album_id}/tracks/',limit=limit,offset=offset,market=market)

    def current_user_playlists(self, limit=50, offset=0):
        return self._call('GET', 'me/playlists', limit=limit, offset=offset)
    
    def user_playlist_create(self, user, pname, public=True, collaborative=False, description=''):
        logger.info("UNIMPLEMENTED")
        pass



def add_all_apps():
    '''
    to be run in ipython3
    make sure apps is populated with valid app information
    from spotify_request import *
    '''
    import spotipy
    import shutil
    import os
    scope = [ 'user-read-private', 'user-follow-read', 'user-library-read', 'user-top-read', 'user-read-recently-played', 'playlist-modify-private', 'playlist-read-private', 'playlist-modify-public' ]
    scope = ' '.join(scope)
    try:
        os.remove('.cache')
    except:
        pass

    for dic in apps:
        client_id = dic['id']
        secret = dic['secret']
        redirect = 'https://jasvandy.github.io/'
        print(f"Name: {dic['name']}")

        fname = f".{dic['name']}_cache"

        soa = spotipy.oauth2.SpotifyOAuth(client_id=client_id, 
            client_secret=secret, redirect_uri=redirect, 
            scope=scope)
        soa.get_access_token()

        shutil.copy2('.cache', fname) # src, dst
        os.remove('.cache')
        
        dic['file'] = fname

