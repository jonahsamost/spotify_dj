import logging
logger = logging.getLogger('Spotify')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

from datetime import datetime, timedelta
from dateutil import parser
import pickle
import math
import unicodedata
import re
import pandas as pd
import time
import numpy as np
import random
from multiprocessing.pool import ThreadPool
import threading
# from multiprocessing import Lock
from threading import Lock

import praw,configparser,json,ast,threading,pickle,math
from spotipy.client import SpotifyException
import spotipy.util as util
import spotipy
import thewire
from spotify_request import SpotifyRequest

from data_types import Item, Terms
from backend import SqlCols

from db import MyDB
from spotify_request import SpotifyRequest

class Spotify(MyDB):
    def __init__(self,username):
        super(Spotify, self).__init__()
        self.username = username
        self.sp = SpotifyRequest(self.username)

        self.sp_lock = Lock()
        self.is_started = 0 
        self.reinit()

        # config settings
        self._top_artist_cnt  = 35
        self._top_song_cnt    = 50
        self._liked_songs_cnt = 30

        self.now = datetime.now().date()
        self._cut_off_date = self.now - timedelta(weeks=5)

        self._top_artists    = None
        self._top_songs_s    = None
        self._top_songs_m    = None
        self._liked_songs    = None
        self._plist_songs    = None
        self._user           = None

        # getters
        self._following = None
        self._top_artists = None

        self.spotify_tbl   = 'spotify_tracks'
        self.spotify_arts  = 'spotify_artists'
        self.spotify_songs = 'spotify_songs'
        self.spotify_empty = 'spotify_empty'
        self.db_create_tables()

    def _reinit(self):
        logger.info("Starting reinit spotify thread")

        while 1:
            # self.token = util.prompt_for_user_token(username, get_scope, client_id, secret, redirect)
            self.sp.reinit()
            self.is_started = 1
            time.sleep(60 * 10)

    def reinit(self):
        t = threading.Thread(target=self._reinit, args=())
        t.start()
        while not self.is_started:
            continue


    def db_create_tables(self):
        q = f'''
        create table if not exists `{self.spotify_arts}` (
        {SqlCols.prim.value} INT NOT NULL AUTO_INCREMENT,
        {SqlCols.artist.value} varchar(100) character set utf16 default null,
        {SqlCols.art_id.value} varchar(100) character set utf16 default null ,
        PRIMARY KEY ({SqlCols.prim.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n',' ')
        self.exec_query(q)

        q = f'''
        create table if not exists `{self.spotify_tbl}` (
        {SqlCols.prim.value} INT NOT NULL AUTO_INCREMENT,
        {SqlCols.art_id.value} varchar(100) character set utf16 default null,
        {SqlCols.song.value} varchar(100) character set utf16 default null,
        {SqlCols.song_id.value} varchar(100) character set utf16 default null, 
        PRIMARY KEY ({SqlCols.prim.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n',' ')
        self.exec_query(q)

        q = f'''
        create table if not exists `{self.spotify_songs}` (
        {SqlCols.song_id.value} varchar(35) character set utf8 , 
        {SqlCols.dance.value} NUMERIC(19,5) , 
        {SqlCols.energy.value} NUMERIC(19,5) , 
        {SqlCols.key.value} NUMERIC(19,5) , 
        {SqlCols.loud.value} NUMERIC(19,5) , 
        {SqlCols.mode.value} NUMERIC(19,5) , 
        {SqlCols.speech.value} NUMERIC(19,5) , 
        {SqlCols.acoust.value} NUMERIC(19,5) , 
        {SqlCols.instru.value} NUMERIC(19,5) , 
        {SqlCols.live.value} NUMERIC(19,5) , 
        {SqlCols.valen.value} NUMERIC(19,5) , 
        {SqlCols.tempo.value} NUMERIC(19,5) , 
        PRIMARY KEY ({SqlCols.song_id.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n',' ')
        self.exec_query(q)

        q = f'''
        create table if not exists `{self.spotify_empty}` (
        {SqlCols.prim.value} INT NOT NULL AUTO_INCREMENT,
        {SqlCols.artist.value} varchar(100) character set utf16 default null,
        {SqlCols.song.value} varchar(100) character set utf16 default null, 
        PRIMARY KEY ({SqlCols.prim.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n',' ')
        self.exec_query(q)


    def db_add_artist(self, artist, artist_id):
        q = f'''
        select * from {self.spotify_arts}
        where {SqlCols.art_id.value} = "{artist_id}"
        '''.strip().replace('\n',' ')
        # df = pd.read_sql(q, self.conn)
        df = self.read_sql_with_lock(q)
        if not df.empty:
            # already added
            return 

        cur = {}
        cur[SqlCols.artist.value] = artist
        cur[SqlCols.art_id.value] = artist_id

        ids_df = pd.DataFrame([cur], columns=cur.keys())
        # ids_df.to_sql(self.spotify_arts, self.engine, if_exists='append', index=False) 
        self.to_sql_with_lock(ids_df, self.spotify_arts)

    @property
    def limit(self):
        return 50

    @property
    def current_user(self):
        '''
        Get detailed profile information about the current user (including the current user's username).
        '''
        if self._user is None:
            self.sp_lock.acquire()
            self._user = self.sp.current_user()
            self.sp_lock.release()
        return self._user

    @property
    def is_inited(self):
        return self.token is not None

    @property
    def following(self):
        '''
        Get the current user's followed artists.
        '''
        if self._following:
            return self._following

        offset = 0
        items = []
        while 1:
            try:
                self.sp_lock.acquire()
                cur = self.sp.current_user_followed_artists(limit=self.limit, after=offset) 
                self.sp_lock.release()
            except Exception as e:
                logger.info(f'Exception: {e}')
                return None

            cur_items = cur['artists']['items']
            len_items = len(cur_items)
            if len_items:
                items += cur_items
                offset += len_items
                if len_items < self.limit:
                    break
            else:
                break

        self._following = [Item(it) for it in items]
        return self._following

    def top_artists(self, term:Terms = Terms.short_term):
        '''
        Get the current user's top artists or tracks based on calculated affinity.
        '''
        if self._top_artists is not None:
            return self._top_artists

        offset = 0
        items = []
        while 1:
            try:
                self.sp_lock.acquire()
                cur = self.sp.current_user_top_artists(limit=self.limit, offset=offset, time_range=term.value) 
                self.sp_lock.release()
            except Exception as e:
                logger.info(f'Exception: {e}')
                return None

            cur_items = cur['items']
            len_items = len(cur_items)
            if len_items:
                items += cur_items
                offset += len_items
                if offset >= self._top_artist_cnt:
                    break
            elif not len_items and offset == 0 and term != Terms.medium_term: # check medium term, no short term music
                term = Terms.medium_term
                continue
            else:
                break

        tas = []
        for i,it in enumerate(items[:self._top_artist_cnt]):
            ta = Item(it)
            ta.user_popularity = i
            tas.append(ta)

        self._top_artists = tas
        return self._top_artists

    def song_ids_from_Items(self, items):
        return list(set([song.id for song in items]))

    def top_songs(self, term:Terms = Terms.short_term):
        '''
        Get the current user's top songs or tracks based on calculated affinity.
        '''
        if self._top_songs_s is not None and term == Terms.short_term:
            return self._top_songs_s
        elif self._top_songs_m is not None and term == Terms.short_term:
            return self._top_songs_m

        offset = 0
        items = []
        while 1:
            try:
                self.sp_lock.acquire()
                cur = self.sp.current_user_top_tracks(limit=self.limit, offset=offset, time_range=term.value) 
                self.sp_lock.release()
            except Exception as e:
                logger.info(f'Exception: {e}')
                return None

            cur_items = cur['items']
            len_items = len(cur_items)
            if len_items:
                items += cur_items
                offset += len_items
                if offset >= self._top_song_cnt:
                    break
            # check medium term, no short term songs
            elif not len_items and offset == 0 and term != Terms.medium_term: 
                term = Terms.medium_term
                continue
            else:
                break

        tas = []
        for i,it in enumerate(items[:self._top_song_cnt]):
            ta = Item(it)
            ta.user_popularity = i
            tas.append(ta)

        if term == Terms.short_term:
            self._top_songs_s = tas
            return self._top_songs_s
        elif term == Terms.medium_term:
            self._top_songs_m = tas
            return self._top_songs_m
        else:
            assert False, 'DO SHIT FOR LONG TERM'

    @property
    def liked_songs(self):
        '''
        Gets a list of the tracks saved in the current authorized user’s “Your Music” library
        '''
        if self._liked_songs is not None:
            return self._liked_songs

        offset = 0
        items = []
        while 1:
            try:
                self.sp_lock.acquire()
                cur = self.sp.current_user_saved_tracks(limit=self.limit, offset=offset)
                self.sp_lock.release()
            except Exception as e:
                logger.info(f'Exception: {e}')
                return None

            cur_items = cur['items']

            len_items = len(cur_items)
            if len_items:
                items += cur_items
                offset += len_items
                if offset >= self._liked_songs_cnt:
                    break
            else:
                break

        tas = []
        for i,it in enumerate(items[:self._liked_songs_cnt]):
            added = parser.parse(it['added_at']).date()
            if added >= self._cut_off_date:
                ta = Item(it['track'])
                ta.user_popularity = i
                tas.append(ta)
            else:
                break

        self._liked_songs = tas
        return self._liked_songs

    @property
    def playlist_songs(self):
        '''
        Get full details of the tracks of all
        playlists that you are the owner of.
        '''
        if self._plist_songs is not None:
            return self._plist_songs

        pids = []
        cid = self.current_user['id']
        self.sp_lock.acquire()
        ps = self.sp.user_playlists(cid , limit=self.limit)
        self.sp_lock.release()
        for p in ps['items']:
            if p['owner']['id'] == self.current_user['id']: 
                pids.append((p['id'] , p['name'], p['tracks']['total']))

        if not pids:
            return 

        psongs = []
        for pid,pname,ptotal in pids:
            # check 50 most recent songs from each playlist
            off_max = max(0,ptotal-self.limit)
            self.sp_lock.acquire()
            pitems = self.sp.playlist_items(pid, limit=self._top_song_cnt, offset=off_max)
            self.sp_lock.release()

            for item in pitems['items']:
                added = parser.parse(item['added_at']).date()
                if added >= self._cut_off_date:
                    it = Item(item['track'])
                    it.date = added # to weight more recently added songs higher
                    psongs.append(it)

        self._plist_songs = psongs
        return self._plist_songs 

    def artist_for_artist_id(self, aid):
        pass


    # GENRE-CENTRIC CODE

    def g_weight(self,i):
        '''
        calculates genre weight
        '''
        return 1/math.sqrt(i+1)

    def get_playlist_songs_genres(self, psongs: [Item]):

        def weight(date):
            return 5 * (1/(1 + (self.now - date).days))

        artists = {}
        for i,s in enumerate(psongs):
            for a in s.artists:
                if a.id not in artists:
                    artists[a.id] = weight(s.date)
                else:
                    artists[a.id] += weight(s.date)
         
        ids = list(artists.keys())
        arts = []
        for i in range(1 + len(ids)//self.limit):
            cur_ids = ids[i*self.limit : i*self.limit + self.limit]
            if not cur_ids:
                break
            self.sp_lock.acquire()
            a = self.sp.artists(cur_ids)
            self.sp_lock.release()
            if 'artists' not in a:
                return None
            arts += a['artists']

        genres = {}
        for i,a in enumerate(arts):
            for g in a['genres']:
                if g in genres:
                    genres[g]['cnt'] += 1
                    genres[g]['weight'] += artists[a['id']]
                else:
                    genres[g] = {'cnt':1, 'weight':artists[a['id']]}

        return genres

    def get_liked_songs_genres(self, lsongs: [Item]):
        if lsongs:
            return self.get_songs_genres(lsongs)
        else:
            return []

    def get_following_genres(self, follow:[Item]):

        ids = [f.id for f in follow]
        arts = []
        for i in range(1 + len(ids)//50):
            cur_ids = ids[i*50 : i*50 + 50]
            self.sp_lock.acquire()
            a = self.sp.artists(cur_ids)
            self.sp_lock.release()
            if 'artists' not in a:
                return None
            arts += a['artists']

        genres = {}
        for i,a in enumerate(arts):
            for g in a['genres']:
                if g in genres:
                    genres[g]['cnt'] += 1
                    genres[g]['weight'] += 1
                else:
                    genres[g] = {'cnt':1, 'weight':1}

        # try to normalize to 5.0
        maxw = 0
        for k,v in genres.items():
            if v['weight'] > maxw: maxw = v['weight']

        for k,v in genres.items():
            v['weight'] /= (maxw / 5)

        return genres

    def get_artists_genres(self, artists:[Item]):
        ids = [a.id for a in artists]
        arts = []
        for i in range(1 + len(ids)//self.limit):
            cur_ids = ids[i*self.limit : i*self.limit + self.limit]
            if not cur_ids:
                break
            self.sp_lock.acquire()
            a = self.sp.artists(cur_ids)
            self.sp_lock.release()
            if 'artists' not in a:
                return None
            arts += a['artists']

        genres = {}
        for i,a in enumerate(arts):
            for g in a['genres']:
                if g in genres:
                    genres[g]['cnt'] += 1
                    genres[g]['weight'] += self.g_weight(i)
                else:
                    genres[g] = {'cnt':1, 'weight':self.g_weight(i)}
        return genres

    def get_songs_genres(self, songs:[Item]):

        # first go through songs, getting artists, making sure to keep track
        # of their location in preference
        artists = {}
        for i,s in enumerate(songs):
            for a in s.artists:
                if a.id not in artists:
                    artists[a.id] = self.g_weight(i)
                else:
                    artists[a.id] += self.g_weight(i)

        ids = list(set([a.id for s in songs for a in s.artists]))
        arts = []
        for i in range(1 + len(ids)//self.limit):
            cur_ids = ids[i*self.limit : i*self.limit + self.limit]
            if not cur_ids:
                break
            self.sp_lock.acquire()
            a = self.sp.artists(cur_ids)
            self.sp_lock.release()
            if 'artists' not in a:
                return None
            arts += a['artists']

        genres = {}
        for i,a in enumerate(arts):
            for g in a['genres']:
                if g in genres:
                    genres[g]['cnt'] += 1
                    genres[g]['weight'] += artists[a['id']]
                else:
                    genres[g] = {'cnt':1, 'weight':artists[a['id']]}

        return genres

    ##########################
    ### LOOKING FOR TRACKS ###
    ##########################

    def spotify_song_id_for_db_ids(self, sids):
        tmp = []
        for sid in sids:
            s = f'({SqlCols.prim.value} = {sid})'
            tmp.append(s)
        tmp = ' or '.join(tmp)

        q = f'''
        select * from {self.spotify_tbl}
        where {tmp}
        '''.strip().replace('\n',' ')
        # df = pd.read_sql(q, self.conn)
        df = self.read_sql_with_lock(q)
        return list(df[SqlCols.song_id.value])

    def song_ids_for_spotify_db_ids(self, tracks):
        '''
        with tracks you get -- say, from mixesdb.artists_get_tracks -- 
        look up in db spotify song ids associated with db song id
        '''
        for i,tr in tracks.iterrows():
            sids = tr[SqlCols.song_id.value]
            ssids = ''
            if sids:
                sids = sids.split('|')
                ssids = self.spotify_song_id_for_db_ids(sids)
                ssids = '|'.join(ssids)
            tracks.loc[i, SqlCols.spot_sid.value] = ssids

    def set_spotify_id_for_tracks(self, tracks, tbl_name=None):
        procs = 10
        # .map() doesnt work on dataframe, so make df into list of smaller dfs
        track_arr = np.array_split(tracks, procs)
        # pool = ThreadPool(processes=procs)
        # pool.map(self._set_spotify_id_for_tracks, track_arr)
        threads = []
        for ta in track_arr:
            t = threading.Thread(target=self._set_spotify_id_for_tracks, args=(ta, tbl_name))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        trks = pd.DataFrame()
        for tr in track_arr:
            trks = trks.append(tr)
        # locs = [l for loc in changed for l in loc]
        return trks

    def _set_spotify_id_for_tracks(self, tracks, tbl_name):

        def _update_tbl(tbl_name, prim , ids):
            q = f'''update {tbl_name} 
            set {SqlCols.song_id.value} = "{ids}"
            where {SqlCols.prim.value} = "{prim}"
            and {SqlCols.song_id.value} is NULL 
            '''.strip().replace('\n',' ')
            logger.info(f"select * from mdb_tracks where id={prim} | {ids=}")
            self.exec_query(q)

        '''
        for a given dataframe of [artist,song], return the id from the
        `spotify_tracks` table NOT the spotify song_id
        in a new column of the dataframe labeled `song_id`
        note: return the id because its shorter 
        '''
        # changed = []
        for i,tr in tracks.iterrows():
            if tr[SqlCols.song_id.value] is not None:
                continue

            artist = tr[SqlCols.artist.value] 
            song   = tr[SqlCols.song.value] 
            if (not artist or not song or song == '?') and tbl_name:
                logger.info(f"select * from mdb_tracks where id={tr['id']}")
                _update_tbl(tbl_name, tr['id'], '')
                continue

            logger.info(f"Searching: {i} : `{artist}`:`{song}`")

            song_df = self.song_search_by_artist_song(artist,song)
            if song_df is None or song_df.empty:
                ids = ''
            else:
                ids = '|'.join([str(i) for i in list(song_df.id)])
                i = 0 
                while len(ids) >= 100:
                    if i > 10: 
                        break
                    ids = ids[: ids.rfind('|')]
                    i+=1
                ids = ids[:100]

            logger.info(f"After searching: `{artist}`:`{song}`")

            # update df and actual table
            tracks.loc[i,SqlCols.song_id.value] = ids

            if tbl_name:
                # set {mixes_db == tr['id']} song_id to {ids}
                _update_tbl(tbl_name, tr['id'], ids)
                logger.info(f"select * from mdb_tracks where id={tr['id']} | {ids=}")

        logger.info("DONE") 

    def get_song_id(self, song, artist_id):
        ''' given song name and artist_id, return song_id '''

        q = f'''
        select * from {self.spotify_tbl}
        where {SqlCols.art_id.value} = "{artist_id}"
        '''.strip().replace('\n',' ')
        # song_df = pd.read_sql(q, self.conn)
        song_df = self.read_sql_with_lock(q)
        if not song_df.empty:
            for i,r in song_df.iterrows():
                if not self.parse_song(r.song, song):
                    song_df.drop([i], inplace=True)

            return song_df

        # get all albums from artist
        self.sp_lock.acquire()
        result = self.sp.artist_albums(artist_id)
        self.sp_lock.release()

        if result['total']:
            ids = [alb['id'] for alb in result['items']]
        else:
            return None

        # get all songs from all albums
        songs = []
        self.sp_lock.acquire()
        albums = self.sp.albums(ids)
        self.sp_lock.release()
        albums = albums['albums']
        for al in albums:
            for tr in al['tracks']['items']:
                cur = {}
                cur[SqlCols.art_id.value] = artist_id
                cur[SqlCols.song.value] = tr['name'][:100]
                cur[SqlCols.song_id.value] = tr['id']
                songs.append(cur)
    
        song_df = pd.DataFrame(songs, columns=songs[0].keys())
        # song_df.to_sql(self.spotify_tbl, self.engine, if_exists='append', index=False) 
        self.to_sql_with_lock(song_df, self.spotify_tbl)
        # song_df = pd.read_sql(q, self.conn)
        song_df = self.read_sql_with_lock(q)

        for i,r in song_df.iterrows():
            if not self.parse_song(r.song, song):
                song_df.drop([i], inplace=True)

        return song_df

    def song_search_by_artist_song(self, artist, song):
        '''
        Find a targetted artist and song
        '''
        arti = self.remove_accents(artist)
        songi = self.remove_accents(song)

        # fast path check if already in db
        q = f'select * from {self.spotify_arts} where {SqlCols.artist.value} = "{artist}"'
        df = self.read_sql_with_lock(q)
        if df is None:
            return None
        if not df.empty:
            aids = list(df[SqlCols.art_id.value])
            trks = []
            for aid in aids:
                tmp = f'({SqlCols.art_id.value} = "{aid}")' 
                trks.append(tmp)
            tmp = ' or '.join(i for i in trks)

            q = f'select * from {self.spotify_tbl} where {tmp}'
            df = self.read_sql_with_lock(q)
            df = df[df[SqlCols.song.value]
                .str.lower().str.find(songi.lower()) == 0] 
            if not df.empty:
                logger.info("Found in SpotifyTracks")
                return df
        
        # clip song/artist cause thats the max db space
        csong = songi[:100] ; cart = arti[:100]
        q = f'''select * from {self.spotify_empty} where 
        ({SqlCols.artist.value} = "{cart}") and ({SqlCols.song.value} = "{csong}")
        '''.strip().replace('\n',' ')
        df = self.read_sql_with_lock(q)
        if df is None:
            return None
        if not df.empty:
            logger.info("Found in SpotifyEmpty")
            return None

        query = f'"{artist}" "{song}"'    
        self.sp_lock.acquire()
        #logger.info(f'LOCK: {artist}:{song}')
        result = self.sp.search(q=query , type='track', limit = self.limit)
        self.sp_lock.release()
        #logger.info(f'UNLOCK: {artist}:{song}')
        tracks = result['tracks']
        if not tracks['total']:
            logger.info(f"Not Found: `{artist}`:`{song}`")
            # cant find song, add to non-present db
            cur = {}
            cur[SqlCols.artist.value] = cart
            cur[SqlCols.song.value]   = csong
            df = pd.DataFrame([cur], columns=cur.keys())
            self.to_sql_with_lock(df, self.spotify_empty)
            return None

        logger.info(f"+++ Found: `{artist}`:`{song}`")

        songs = []
        for tr in tracks['items']:
            arts = []
            namei = self.remove_accents(tr['name'])
            if namei.lower().find(songi.lower()) != 0:
                continue

            for art in tr['artists']:
                aname = art['name']
                aname = self.remove_accents(aname)[:100]
                if aname.lower().find(arti.lower()) == -1:
                    cur = {}
                    cur[SqlCols.art_id.value] = art['id']
                    cur[SqlCols.song.value] = tr['name'][:100]
                    cur[SqlCols.song_id.value] = tr['id']
                    songs.append(cur)
                    self.db_add_artist(aname, art['id'])

        if not songs:
            logger.info(f"Not Found: `{artist}`:`{song}`")
            # cant find song, add to non-present db
            cur = {}
            cur[SqlCols.artist.value] = cart
            cur[SqlCols.song.value]   = csong
            df = pd.DataFrame([cur], columns=cur.keys())
            self.to_sql_with_lock(df, self.spotify_empty)
            return None

        song_df = pd.DataFrame(songs, columns=songs[0].keys())
        # dont double add songs if they have same artist id
        idxs = list(song_df[[SqlCols.art_id.value,SqlCols.song.value]].drop_duplicates().index)
        song_df = song_df.loc[song_df.index[idxs]]
        self.to_sql_with_lock(song_df, self.spotify_tbl)
        q = f'select * from {self.spotify_tbl} where '
        trks = []
        for s in songs:
            tmp = f'({SqlCols.song_id.value} = "{s[SqlCols.song_id.value]}")' 
            trks.append(tmp)
        q += ' or '.join(trks)  

        return self.read_sql_with_lock(q)

    def remove_accents(self, input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def parse_song(self, sng, song):
        if not sng: return False
        sng = sng.split(' - ')[0]
        sng = re.sub("[\(\[].*?[\)\]]", "", sng)
        sng = self.remove_accents(sng)
        sng = sng.strip()
        return sng == song


    def artist_id_from_name(self, artist):
        ''' 
        given artist name, return potential artist ids
        if artist is not found in db, add artist to db
        '''
        logger.info(f"{artist}")
        
        query = f'''
        select {SqlCols.art_id.value}, {SqlCols.artist.value} from `{self.spotify_arts}` 
        where {SqlCols.artist.value} = \"{artist}\"
        '''.strip().replace('\n','')
        # exists = pd.read_sql(query, self.conn)
        exists = self.read_sql_with_lock(query)
        if not exists.empty:
            return list(exists[SqlCols.art_id.value].drop_duplicates())

        off = 0
        ids = []
        while 1:
            try:
                self.sp_lock.acquire()
                logger.info("LOCK: AID FROM NAME")
                result = self.sp.search(q=f'artist:{artist}' , type='artist', limit = self.limit , offset = off * self.limit)
                self.sp_lock.release()
                logger.info("UNLOCK: AID FROM NAME")
            except:
                break

            items = result['artists']['items']
            if not items:
                break
            for item in items:
                if item['name'].lower() == artist.lower():
                    cur = {}
                    cur[SqlCols.artist.value] = item['name'][:100]
                    cur[SqlCols.art_id.value] = item['id']
                    ids.append(cur)
            if len(items) < self.limit:
                break 
            off += 1

            if off == 5:
                break

        found = 1
        if not ids:
            found = 0 
            logger.info(f"Could not find {artist} in Spotify")
            cur = {}
            cur[SqlCols.artist.value] = artist[:100]
            cur[SqlCols.art_id.value] = ''
            ids.append(cur)
        else:
            logger.info(f"Found {artist} in Spotify")

        ids_df = pd.DataFrame(ids, columns=ids[0].keys())
        # ids_df.to_sql(self.spotify_arts, self.engine, if_exists='append', index=False) 
        self.to_sql_with_lock(ids_df, self.spotify_arts)
        if found:
            return list(ids_df[SqlCols.art_id.value].drop_duplicates())
        else: 
            return []
    
    def unsplit_song_ids(self, df, one=False):
        '''
        return all spotify song ids
        after self.song_ids_for_spotify_db_ids is called,
        which sets 'spot_sid' on the dataframe
        '''
        ssids = list(df[SqlCols.spot_sid.value])
        songids = []
        for s in ssids:
            if not s: 
                continue
            ss = s.split('|')
            if one:
                songids += [random.choice(ss)]
            else:
                songids += ss
        return list(set(songids))

    def audio_features_to_dic(self, track):
        cur = {}
        cur[SqlCols.song_id.value] =  track[SqlCols.prim.value].strip()
        cur[SqlCols.dance.value]   =  track[SqlCols.dance.value]
        cur[SqlCols.energy.value]  =  track[SqlCols.energy.value]
        # cur[SqlCols.key.value]     =  track[SqlCols.key.value]
        cur[SqlCols.key.value]     =  track['key']
        cur[SqlCols.loud.value]    =  track[SqlCols.loud.value]
        cur[SqlCols.mode.value]    =  track[SqlCols.mode.value]
        cur[SqlCols.speech.value]  =  track[SqlCols.speech.value]
        cur[SqlCols.acoust.value]  =  track[SqlCols.acoust.value]
        cur[SqlCols.instru.value]  =  track[SqlCols.instru.value]
        cur[SqlCols.live.value]    =  track[SqlCols.live.value]
        cur[SqlCols.valen.value]   =  track[SqlCols.valen.value]
        cur[SqlCols.tempo.value]   =  track[SqlCols.tempo.value]
        return cur

    def audio_features_for_tracks(self, songids):
        trks = []
        for i in range(0 , len(songids), 100):
            cur_ids = songids[i:i+100]
            cur_trk_ids = {}
            for sid in cur_ids: cur_trk_ids[sid] = 0 

            ssids = ','.join(cur_ids)
            self.sp_lock.acquire()
            tracks = self.sp.audio_features(tracks=ssids)
            self.sp_lock.release()
            if not tracks:
                continue
            for track in tracks:
                if track:
                    cur_trk_ids[track['id']] = 1
                    trks.append(self.audio_features_to_dic(track))

            # some songs dont have audio features ... 
            for sid in [k for k,v in cur_trk_ids.items() if v == 0]:
                cur = {}
                cur[SqlCols.song_id.value] = sid
                trks.append(cur)
            
        if trks:
            df = pd.DataFrame(trks, columns=trks[0].keys())
            df = df.round(5)
            df = df.drop_duplicates(subset=SqlCols.song_id.value)
            return df
        return None

    def db_add_spotify_songs(self, songids):
        q = f'select {SqlCols.song_id.value} from {self.spotify_songs}'
        cur_songs = pd.read_sql(q, self.conn)

        # only look for songs we dont have in db
        si_df = pd.DataFrame(songids, columns=[SqlCols.song_id.value])
        whatsin = si_df[SqlCols.song_id.value].isin(list(cur_songs.song_id)) 
        toadd = si_df[~whatsin]
        sids = list(toadd[SqlCols.song_id.value])

        df  = self.audio_features_for_tracks(sids)
        if df is None:
            return None

        indb = df[SqlCols.song_id.value].isin(list(cur_songs.song_id))
        toadd = df[~indb]
        # toadd.to_sql(self.spotify_songs, self.engine, if_exists='append', index=False) 
        self.to_sql_with_lock(toadd, self.spotify_songs)

    def db_spotify_songs_for_sids(self, sids):
        df = pd.DataFrame()
        for i in range(0, len(sids), 100):
            t = []
            for s in sids[i : i + 100]:
                t.append(f'({SqlCols.song_id.value} = "{s}")')
            t = ' or '.join(t)

            q = f'''
            select * from {self.spotify_songs}
            where {t}
            '''.strip().replace('\n',' ')
            df = df.append(pd.read_sql(q, self.conn))
        return df
        

    ###########################
    ###########################
    ###########################
    ##### END CURRENT CODE ####
    ###########################
    ###########################
    ###########################


    # ''' given list of dictionaries, get song ids 'nd unkown artists,songs '''
    def thread_get_spotify_songs(self, dic_list, loc):
        unk_artists = []
        unk_songs = []
        songs = []
        for dic in dic_list:
            if self.year_cutoff:
                if dic['year'] > self.year_cutoff:
                    continue
            print(dic)
            artid = self.get_artist_id(dic['artist'])
            print('{}:{}'.format(dic['artist'], artid))
            if not artid: 
                unk_artists.append(dic['artist'])
                continue

            songid,track_info = self.get_song_id(dic['song'].lower(),artid)
            if not songid:
                songid,track_info = self.artist_get_specific_track(artid,dic['song'].lower())
                if not songid:
                    unk_songs.append({'artist':dic['artist'] , 'artist_id':artid , 'song':dic['song']})
                    continue
                else:
                    if dic['year'] == -1:
                        if thewire.get_year_from_item(track_info) >= self.year_cutoff: 
                            continue 
            else:
                if dic['year'] == -1:
                    if thewire.get_year_from_item(track_info) >= self.year_cutoff: 
                        continue 


            songs.append(songid)

        self.song_retrieve[loc] = {'songs':songs,'unk_artists':unk_artists,'unk_songs':unk_songs}


    def get_song_id_match_artist_name(self, song, artist_name):
        ''' tries to get song id matching on name of artist'''
        off = 0
        while 1:
            result = None
            try:
                self.sp_lock.acquire()
                result = self.sp.search(q = song , type = 'track', limit = self.limit , offset = off*self.limit)
                self.sp_lock.release()
            except:
                return None
            items = result['tracks']['items']
            for item in items:
                for art in item['artists']:
                    if art['name'].lower().find(artist_name.lower()):
                        return item['id']
            off += 1
            if off == 10:
                break 

        return None

    ''' if you can find the artist but not the track, search through all tracks'''
    def artist_get_specific_track(self,artist_id,search):
        self.sp_lock.acquire()
        alb = self.sp.artist_albums(artist_id, album_type='album,appears_on,single,compilation', country=None, limit=50, offset=0)
        self.sp_lock.release()
        for album in alb['items']:
            album_id = album['id']
            self.sp_lock.acquire()
            tracks = self.sp.album_tracks(album_id,limit=50,offset=0,market=None)
            self.sp_lock.release()
            for track in tracks['items']:
                name = track['name']
                if name.lower().find(search) != -1:
                    return (track['id'],track)
        return (None,None)


    ''' get playlist id for given name, make new one if name doesnt exist'''
    def get_playlist_id(self):
        pname = 'deejay'
        self.sp_lock.acquire()
        pitems = self.sp.current_user_playlists()
        self.sp_lock.release()
        for i in pitems['items']:
            name =  i['name'].encode('ascii', 'ignore')
            if name == pname:
                return i['id']
        self.sp_lock.acquire()
        result = self.sp.user_playlist_create(self.username, pname , public = False)
        self.sp_lock.release()
        return result['id']

    ''' add songlist to playlist'''
    def playlist_add_songs(self, playlist_id,songs):
        songlist = list(set(songs))
        if songlist:
            i = 0 
            while 1:
                try:
                    self.sp_lock.acquire()
                    self.sp.user_playlist_add_tracks(self.username, playlist_id, songlist[i*100: i*100 + 100], position = i*100)
                    self.sp_lock.release()
                    i+=1
                except:
                    return 


