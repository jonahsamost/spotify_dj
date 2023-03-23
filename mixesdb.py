import logging
logger = logging.getLogger('Spotify')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

from backend import Backend, SqlCols
from db import MyDB
from multiprocessing.pool import ThreadPool
from enum import Enum
import re
import random
import threading
import os
import traceback
import pandas as pd
import math
import time
import numpy as np
from mixesdb_genres import mdb_gen_to_spot_gen

class MixesDB(Backend, MyDB):
    def __init__(self, user, artists=None, genres=None, use_threads=True):
        super(MixesDB, self).__init__()
        assert artists is None or type(artists) == list, 'Artists not in list'
        assert genres is None  or type(genres) == list, 'Artists not in list'

        self.user = user 

        self.base_url = 'https://www.mixesdb.com'
        self.artists = artists
        # self.genres  = genres
        self.use_threads = use_threads

        self.max_mix = 25
        self.pg_cnt = 200
        self.tracks = []
        self.genres = None

        self.genres_file = '.mixdb_genres.txt'
        self.artist_file = '.mixdb_artists.txt'

        self.artist_tbl = 'mdb_artists'
        self.mixes_tbl  = 'mdb_tracks' 
        self.genre_tbl  = 'mdb_genres'
        self.genre_mix  = 'mdb_genre_mixes'
        self.mix_urls   = 'mdb_mix_urls'
        self.db_create_tables()

    def run(self, artists, genres):
        '''
        runs per-user 
        '''
        tracks = self.read_sql_with_lock(f'select * from {self.mixes_tbl}')
        # TODO EVERYTHING
        # gens_df = self.genres_get_tracks(genres, tracks)
        # gen_df = self.genres_process_tracks(genres)

        # arts = self.artists_get_tracks(artists)
        # art_df = self.artists_process_tracks(arts)

        # df = gen_df.append(art_df)
        # df.drop_duplicates(inplace=True)

        # trks = self.user.set_spotify_id_for_tracks(df)
        # tracks = trks[trks[SqlCols.song_id.value] != '']
        # tracks = tracks[~tracks[SqlCols.song_id.value].isnull()]

        # return df
        

    def db_create_tables(self):
        '''
        create tables for use in this clas
        '''
        # artist table: pkey_id | artist_name | artist_href | fkey_mixes_table
        q = f'''
        create table if not exists `{self.artist_tbl}` (
        {SqlCols.prim.value} INT NOT NULL AUTO_INCREMENT,
        {SqlCols.name.value} varchar(80) character set utf16,
        {SqlCols.href.value} varchar(200) character set utf8,
        {SqlCols.rel.value} varchar(80) character set utf8,
        PRIMARY KEY ({SqlCols.prim.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n',' ')
        self.exec_query(q)

        q = f'''
        create table if not exists `{self.mix_urls}` (
        {SqlCols.prim.value} INT NOT NULL AUTO_INCREMENT,
        {SqlCols.art_id.value} INT NOT NULL default -1 ,
        {SqlCols.href.value} varchar(200) character set utf8,
        {SqlCols.mix_url.value} varchar(200) character set utf8,
        PRIMARY KEY ({SqlCols.prim.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n',' ')
        self.exec_query(q)

        # genre table: pkey_id | artist_name | artist_href | fkey_mixes_table
        q = f'''
        create table if not exists `{self.genre_tbl}` (
        {SqlCols.prim.value} INT NOT NULL AUTO_INCREMENT,
        {SqlCols.name.value} varchar(80) character set utf16,
        {SqlCols.href.value} varchar(200) character set utf8,
        {SqlCols.count.value} INT default -1 ,
        PRIMARY KEY ({SqlCols.prim.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n',' ')
        self.exec_query(q)

        q = f'''
        create table if not exists `{self.genre_mix}` (
        {SqlCols.prim.value} INT NOT NULL AUTO_INCREMENT,
        {SqlCols.name.value} varchar(80) character set utf16,
        {SqlCols.href.value} varchar(200) character set utf8,
        PRIMARY KEY ({SqlCols.prim.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n',' ')
        self.exec_query(q)

        # mixes table 
        q = f'''
        create table if not exists {self.mixes_tbl} (
        {SqlCols.prim.value} INT NOT NULL AUTO_INCREMENT,
        {SqlCols.art_id.value} INT NOT NULL default -1 ,
        {SqlCols.gen_id.value} INT NOT NULL default -1,
        {SqlCols.set_href.value} varchar(200) character set utf8,
        {SqlCols.artist.value} varchar(100) character set utf16,
        {SqlCols.song.value} varchar(100) character set utf16,
        {SqlCols.song_id.value} varchar(100) character set utf16,
        PRIMARY KEY ({SqlCols.prim.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n',' ')
        self.exec_query(q)
        self.commit_with_lock()

    def db_update_genres_table(self, gen_name, gen_url):
        q = f'''
        insert into `{self.genre_tbl}`
        ({SqlCols.name.value}, {SqlCols.href.value}) values
        ("{gen_name}", "{gen_url}")
        '''.strip().replace('\n','')
        return self.exec_query(q)

    def db_update_artists_table(self, art_name, art_href):
        q = f'''
        insert into `{self.artist_tbl}`
        ({SqlCols.name.value}, {SqlCols.href.value}) values
        ("{art_name}", "{art_href}")
        '''.strip().replace('\n','')
        return self.exec_query(q)

    def db_add_relationship(self, art_id, sub_art_ids):
        q = f'''
        update {self.artist_tbl}
        set {SqlCols.rel.value} = {sub_art_ids}
        where {SqlCols.prim.value} = {art_id}
        '''.strip().replace('\n','')
        return self.exec_query(q)

    def db_get_genre_id(self, gen_name, gen_url, make=True):
        q = f"""select * from {self.genre_tbl} 
        where {SqlCols.name.value} = \'{gen_name}\' and 
        {SqlCols.href.value} = \'{gen_url}\'
        """.strip().replace('\n','')

        if make:
            exists = pd.read_sql(q, self.conn)
            if exists.empty:
                self.db_update_genres_table(gen_name, gen_url)
                self.commit_with_lock()

        exists = pd.read_sql(q, self.conn)
        if not make and exists.empty:
            return -1

        assert not exists.empty, 'No genre/url pair found'
        assert len(exists) == 1, 'More than 1 genre found'
        return exists.iloc[0]['id']

    def artist_from_artist(self, artist):
        q = f'''
        select * from {self.artist_tbl}
        where {SqlCols.name.value} like "{artist}"
        '''.strip().replace('\n',' ')
        # df = pd.read_sql(q, self.conn)
        df = self.read_sql_with_lock(q)
        if df.empty:
            # fall back to only matching on first part of artist
            q = f'''
            select * from {self.artist_tbl}
            where {SqlCols.name.value} like "{artist}%"
            '''.strip().replace('\n',' ')
            # df = pd.read_sql(q, self.conn)
            df = self.read_sql_with_lock(q)
            if df.empty:
                logger.info(f"Cant find {artist}")
                return None

            for i, row in df.iterrows():
                name = row[SqlCols.name.value]
                # remove chars between/including [] and ()
                tmp = re.sub("[\(\[].*?[\)\]]", "", name)
                tmp = tmp.strip()
                if tmp.find(artist) != 0:
                    df.drop(i, inplace=True)
            
            if df.empty:
                logger.info(f"Cant find {artist}")
                return None

        return df

    def tracks_from_artist_href(self, href):
        df = self.artist_from_href(href)
        aid = df[SqlCols.prim.value].iloc[0]
        return self.tracks_from_artist_id(aid)

    def tracks_from_artist_ids(self, aids):
        '''
        get all tracks from mixes table that have certain artist ids
        '''
        tmp = []
        for aid in aids:
            s = f'({SqlCols.art_id.value} = {aid})'
            tmp.append(s)
        tmp = ' or '.join(tmp)

        q = f'''
        select * from {self.mixes_tbl}
        where {tmp}
        '''.strip().replace('\n',' ')
        return pd.read_sql(q, self.conn)

    def tracks_from_artist_id(self, aid):
        return self.tracks_from_artist_ids([aid])

    def relations_from_artist_id(self, aid):
        q = f'''
        select * from {self.artist_tbl} 
        where {SqlCols.prim.value} = {aid}
        '''.strip().replace('\n',' ')
        df = pd.read_sql(q, self.conn)
        if not df.empty:
            rel = df.iloc[0][SqlCols.rel.value]
            if rel:
                return rel.split('|')
        return []

    def all_relations_from_artist_id(self, aid):
        rels = [str(aid)]
        aid_rels = self.relations_from_artist_id(aid)
        rels += aid_rels
        for ai in aid_rels:
            rels += self.relations_from_artist_id(ai)
        return list(set(rels))

    def artist_from_relation(self, rel_id):
        q = f'''
        select * from {self.artist_tbl}
        where {SqlCols.prim.value} = {rel_id}
        '''.strip().replace('\n',' ')
        df = pd.read_sql(q, self.conn)
        name = df.iloc[0][SqlCols.name.value]
        href = df.iloc[0][SqlCols.href.value]
        return (name,href)

    def relations_from_artist_href(self, href):
        df = self.artist_from_href(href)
        if df is None:
            return None
        rels = df.iloc[0][SqlCols.rel.value]
        if rels:
            return rels.split('|')
        return []

    def artist_from_href(self, href):
        q = f'''
        select * from {self.artist_tbl}
        where {SqlCols.href.value} = "{href}"
        '''.strip().replace('\n',' ')
        # df = pd.read_sql(q, self.conn)
        df = self.read_sql_with_lock(q)
        if df.empty:
            return None
        return df

    def db_get_artist_id(self, art_name, art_href, make=True):

        q = f"""select * from {self.artist_tbl} 
        where {SqlCols.name.value} = \'{art_name}\' and 
        {SqlCols.href.value} = \'{art_href}\'
        """.strip().replace('\n',' ')

        if make:
            exists = pd.read_sql(q, self.conn)
            if exists.empty:
                self.db_update_artists_table(art_name, art_href)
                self.commit_with_lock()

        exists = pd.read_sql(q, self.conn)
        if not make and exists.empty:
            return -1

        assert not exists.empty, 'No artist/href pair found'
        assert len(exists) == 1, 'More than 1 artist/href pair found'
        return exists.iloc[0]['id']

    def db_get_artist_tracks(self, art_id):
        q = f'''
        select * from {self.mixes_tbl} 
        where {SqlCols.art_id.value} = {art_id}
        '''.strip().replace('\n','')
        return pd.read_sql(q, self.conn)

    def db_add_genre_sets(self, mixes):

        hrs = []
        df = pd.DataFrame()
        for i in range(1 + len(mixes)//50):
            cur_hrefs = mixes[i*50 : i*50 + 50]
            if not cur_hrefs:
                continue
            t = []
            for h in cur_hrefs:
                q = f'({SqlCols.set_href.value} = "{h}")'
                t.append(q)
            t = ' or '.join(t)

            q = f'''
            select * from {self.mixes_tbl} where {t}
            '''.strip().replace('\n',' ')
            df = df.append(self.read_sql_with_lock(q))

        sas = df[[SqlCols.artist.value, SqlCols.song.value]]
        sas = sas.drop_duplicates()
        self.user.set_spotify_id_for_tracks(sas)

    def _db_add_genre_sets(self, gen_name, gen_url, href_set_list):
        if not href_set_list: return None

        gen_id = self.db_get_genre_id(gen_name, gen_url)

        q = f'''select * from {self.mixes_tbl} where 
        {SqlCols.gen_id.value} = {gen_id} 
        '''.strip().replace('\n','')
        gen_df = pd.read_sql(q, self.conn)

        mixes = []
        for mix in href_set_list:
            ls = mix['set']
            if not gen_df[gen_df[SqlCols.set_href.value] == mix['href']].empty:
                continue

            cur = {}
            cur[SqlCols.gen_id.value] = gen_id 
            cur[SqlCols.set_href.value] = mix['href']
            if not ls:
                cur[SqlCols.artist.value] = ''
                cur[SqlCols.song.value] = ''
                mixes.append(cur)
            else:
                for a,s in ls:
                    cur[SqlCols.artist.value] = a[:100]
                    cur[SqlCols.song.value] = s[:100]
                    mixes.append(cur)
                    cur = cur.copy()

        self.user.set_spotify_id_for_tracks(mixes)
        if mixes: 
            mix_df = pd.DataFrame(mixes, columns=mixes[0].keys())
            # mix_df.to_sql(self.mixes_tbl, self.engine, if_exists='append', index=False) 
            self.to_sql_with_lock(mix_df, self.mixes_tbl)
            # self.commit_with_lock()
            return mix_df

        return None

    def db_add_artist_sets(self, art_name, art_href, href_set_list):
        if not href_set_list: return None

        art_id = self.db_get_artist_id(art_name, art_href)

        q = f'''select * from {self.mixes_tbl} where 
        {SqlCols.art_id.value} = {art_id} 
        '''.strip().replace('\n','')
        art_df = pd.read_sql(q, self.conn)

        mixes = []
        for mix in href_set_list:
            ls = mix['set']
            # dont double add sets 
            if not art_df[art_df[SqlCols.set_href.value] == mix['href']].empty:
                continue

            cur = {}
            cur[SqlCols.art_id.value] = art_id
            cur[SqlCols.set_href.value] = mix['href']
            if not ls:
                cur[SqlCols.artist.value] = ''
                cur[SqlCols.song.value] = ''
                mixes.append(cur)
            else:
                for a,s in ls:
                    cur[SqlCols.artist.value] = a[:100]
                    cur[SqlCols.song.value] = s[:100]
                    mixes.append(cur)
                    cur = cur.copy()

        self.user.set_spotify_id_for_tracks(mixes)
        if mixes: 
            mix_df = pd.DataFrame(mixes, columns=mixes[0].keys())
            # mix_df.to_sql(self.mixes_tbl, self.engine, if_exists='append', index=False) 
            self.to_sql_with_lock(mix_df, self.mixes_tbl)
            # self.commit_with_lock()


    def artists_get_tracks(self, artists):
        '''
        From artists, get tracks from all of their sets
        '''
        self.artists = artists
        return [self.get_artist_data(a) for a in self.artists]

    def artists_process_tracks(self, df_list):
        df = pd.DataFrame() 
        for o in df_list: 
            if type(o) != pd.DataFrame: 
                continue 
            df = df.append(o[['artist','song','song_id']]) 

        df.drop_duplicates(inplace=True) 
        return df

    def genres_get_genres(self, genres):
        '''
        given Spotify genres, get the associated
        mixes db genres
        '''
        if self.genres:
            return self.genres

        gens = []
        for g in genres:
            for k,v in mdb_gen_to_spot_gen.items():
                if g in v:
                    gens.append(k)
        self.genres = list(set(gens))
        return self.genres


    def genres_get_tracks(self, genres, track_df):
        '''
        for user specified genres, get genre information
        '''
        gens = self.genres_get_genres(genres)
        return self.get_genres_data(gens, track_df)

    def _genres_process_tracks(self, genres):
        gens = self.genres_get_genres(genres)

        q = f'select {SqlCols.href.value} from {self.genre_mix} where'
        tmp = []
        for g in gens:
            tmp.append(f'({SqlCols.name.value} = "{g}")')
        q += ' or '.join(tmp)
        df = pd.read_sql(q, self.conn)
        df.drop_duplicates(inplace=True)
        
        q = f'''select {SqlCols.set_href.value},{SqlCols.artist.value},{SqlCols.song.value}
        from {self.mixes_tbl} 
        '''
        fulldf = pd.read_sql(q,self.conn)
        out = []
        for i,row in df.iterrows():
            cur = fulldf[fulldf[SqlCols.set_href.value] == row[SqlCols.href.value]]
            out.append(cur)


    def genres_process_tracks(self, genres):
        '''
        after you retrieve genre data from mixesdb,
        get desired genre data and return deduped artist/song dataframe
        '''
        logger.info('Processing Tracks ... this takes time')
        gens = self.genres_get_genres(genres)

        q = f'select {SqlCols.href.value} from {self.genre_mix} where'
        tmp = []
        for g in gens:
            tmp.append(f'({SqlCols.name.value} = "{g}")')
        q += ' or '.join(tmp)
        df = pd.read_sql(q, self.conn)
        df.drop_duplicates(inplace=True)

        tmp = []
        for i,row in df.iterrows():
            cur = row[SqlCols.href.value]
            q = f'({SqlCols.set_href.value} = "{cur}")'
            tmp.append(q)

        tracks = pd.DataFrame()
        topq = f'''select {SqlCols.artist.value}, {SqlCols.song.value}, {SqlCols.song_id.value}
        from {self.mixes_tbl} where '''
        for i in range(1 + len(tmp)//300):
            print(f'{i*300} ... {len(tmp)}')
            cur = tmp[i*300 : i*300 + 300]
            if not cur:
                continue
            
            q = topq + ' or '.join(cur)
            tracks = tracks.append(pd.read_sql(q, self.conn))

        tracks.drop_duplicates(inplace=True)

        # TODO
        '''
        remove rows for which song_id is not null 
        '''

        return tracks


    # TODO
    def add_spotify_songid_to_mixes_table(self):
        pass
        '''
        for each row in mdb_tracks:
            if song_id == NULL:
                if (does artist/song pair exist in another row):
                    make all equal rows have same song_id
                else:
                    look up artist/song in spotify
                    add song_id to all rows which match song/artist
        '''

    def get_genres_data(self, genres, track_df):
        '''
        get all tracks associated with genre
        from genre_mix table
        '''
        gen_dfs = []
        for g in genres:
            q = f'''
            select * from {self.genre_mix} 
            where {SqlCols.name.value} = "{g}"
            '''.strip().replace('\n',' ')
            href_df = self.read_sql_with_lock(q)

            gen_df = track_df[track_df.set_href.isin(href_df.href)] 
            gen_dfs.append({'genre':g, 'df':gen_df})

        return gen_dfs

    def genre_collect_mixes_from_pg(self, gen_name, gen_url):
        logger.info(f"Collecting mixes for {gen_name}")

        q = f'''
        select * from {self.genre_mix}
        where {SqlCols.name.value} = "{gen_name}"
        '''.strip().replace('\n',' ')
        gen_df = pd.read_sql(q, self.conn)

        if not gen_df.empty:
            # start at end of dataframe if found
            eurl = gen_df.href.iloc[-1][len(self.base_url) + len('/w/'):]
            url = f'https://www.mixesdb.com/db/index.php?title=Category:{gen_name}&pagefrom={eurl}'
        else:
            url = gen_url

        cnt = 0 
        while 1:
            logger.info(f'{gen_name} : {cnt} : {url}')
            cnt += 1
            soup = self.get_pagex(url)
            lis = soup.find_all('li')
            if not lis:
                break
            mixes = []
            for li in lis:
                a = li.find('a')
                if a is None: 
                    continue
                link = a['href']
                if link.find('/w/') != 0:
                    continue 
                try:
                    # make sure first char is a number (corresponds to year) 
                    int(link[len('/w/')])
                    mixes.append(self.base_url + link)
                except:
                    continue 

            out = []
            for mix in mixes:
                cur = {}
                cur[SqlCols.name.value] = gen_name
                cur[SqlCols.href.value] = mix[:200]
                out.append(cur)

            if out:
                df = pd.DataFrame(out, columns=out[0].keys())
                self.to_sql_with_lock(df, self.genre_mix)

            url = self.nxt_url_from_soup(soup)
            if url is None:
                break

        q = f'''
        select {SqlCols.href.value} from {self.genre_mix}
        where {SqlCols.name.value} = "{gen_name}"
        '''.strip().replace('\n',' ')
        gen_df = pd.read_sql(q, self.conn)
        return list(gen_df[SqlCols.href.value])

    def genre_find_genre(self, genre):
        '''
        get genre from spotify genre
        '''
        logger.info(f"Searching {genre}")
        q = f'''select * from {self.genre_tbl}
        where {SqlCols.name.value} = "{genre}"
        '''.strip().replace('\n',' ')
        df = self.read_sql_with_lock(q)
        if not df.empty:
            r = df.iloc[0]
            return (r[SqlCols.name.value], r[SqlCols.count.value], r[SqlCols.href.value])
        return (None,None,None)

    def refresh_genres(self):
        gen_url = gen_count = gen_name = None
        url ='https://www.mixesdb.com/w/Category:Style' 
        soup = self.get_pagex(url)

        genres = []
        tbl = soup.find('table', {'id':'cellboxtable'})
        for t in tbl:
            if t and t != '\n':
                hrefs = t.find_all('p')
                if not hrefs:
                    continue
                for h in hrefs:
                    boxes = h.find_all('a')
                    if not boxes:
                        continue

                    cur_gens = []
                    for gen in boxes:
                        cur = {}
                        cur[SqlCols.name.value ] = gen.text
                        cur[SqlCols.href.value ] = self.base_url + gen['href']
                        cur[SqlCols.count.value] = 0
                        cur_gens.append(cur)

                    for style in h.text.split('\n'):
                        if not style:
                            break
                        gen_name, cnt = style.split('\xa0')
                        gen_count = int(cnt.strip().replace('(','').replace(')','').replace(',',''))

                        for cg in cur_gens:
                            if cg[SqlCols.name.value] == gen_name:
                                cg[SqlCols.count.value] = gen_count
                                break

                    genres += cur_gens

        self.exec_query('delete from {self.genre_tbl}')
        self.conn.commit()

        df = pd.DataFrame(genres, columns=genres[0].keys())
        self.to_sql_with_lock(df, self.genre_tbl)
                        
        return None

    def spotify_lookup(self, art_tracks):
        ats = pd.DataFrame()
        for at in art_tracks:
            ats = ats.append(at)

        # TODO 
        '''
        if song_id column exists and is not null, 
            remove it cause we've already looked up that song
        '''

        art_songs = ats[[SqlCols.artist.value, SqlCols.song.value]].drop_duplicates()
        out = self.user.set_spotify_id_for_tracks(art_songs)


    def get_artist_data(self, artist):
        '''
        from artist, returns spotify song ids
        '''
        art_id = self.set_artist_data(artist)
        if art_id == -1:
            return []

        ids = self.all_relations_from_artist_id(art_id)
        tracks = self.tracks_from_artist_ids(ids) 

        # TODO 
        '''
        remove songs with ids that are not nll
        '''

        # work that includes Spotify --
        # tracks, track_locs = self.user.set_spotify_id_for_tracks(tracks)
        # self.mixes_tbl_update_song_ids(tracks, track_locs)
        # self.user.song_ids_for_spotify_db_ids(tracks)
        return tracks

    def set_artist_data(self,artist, art_href=None):
        logger.info(f"MixesDB searching for {artist}")

        aga = self.artist_get_artist(artist, art_href)
        if aga is None:
            logger.info(f'MixesDB not found {artist}')
            return -1
        aid,df,rels = aga
        if aid == -1:
            logger.info(f'MixesDB not found {artist}')
            return -1

        if df is not None:
            hrefs = list(df[SqlCols.mix_url.value])
            self.receive_track_info(aid, hrefs)

        for rel in rels:
            name,href = self.artist_from_relation(rel)
            tracks = self.tracks_from_artist_href(href)
            if tracks.empty:
                logger.info(f"MDB relationship {artist} -> {name}")
                self.set_artist_data(name,href)

        return aid

    def receive_tracks_from_href(self, artist_id, href):
        '''
        Given href, attempts to get tracks from mixes that we dont yet have
        and puts those artist/song/hrefs into mixes table
        '''
        logger.info("Receiving track info from hrefs")

        href,tracks = self.mixes_tracklist_for_mix_href(href)
        all_tracks = self.parse_track_on_href(href,tracks)
        self.add_tracks_to_mixes_tbl(artist_id, [all_tracks])

    def add_tracks_to_mixes_tbl(self, artist_id, sets):
        out = []
        for cur in sets:
            href = cur['href']
            tracks = cur['set']
            if tracks:
                for track in tracks:
                    dic = {}
                    dic[SqlCols.set_href.value] = href[:200]
                    dic[SqlCols.artist.value] = track[0][:100]
                    dic[SqlCols.song.value] = track[1][:100]
                    dic[SqlCols.art_id.value] = artist_id
                    out.append(dic)
            else:
                dic = {}
                dic[SqlCols.set_href.value] = href[:200]
                dic[SqlCols.artist.value] = ''
                dic[SqlCols.song.value] = ''
                dic[SqlCols.art_id.value] = artist_id
                out.append(dic)

        if out:
            df = pd.DataFrame(out, columns=out[0].keys())
            self.to_sql_with_lock(df, self.mixes_tbl)

    def mixes_tbl_update_song_ids(self, tracks, track_locs):
        for i,trk in tracks.iterrows():
            l = trk[SqlCols.prim.value]
            if l not in track_locs:
                continue
            val = trk[SqlCols.song_id.value]
            if val is None:
                val = ''

            q = f'''
            update {self.mixes_tbl}
            set {SqlCols.song_id.value} = "{val}"
            where {SqlCols.prim.value} = {trk[SqlCols.prim.value]}
            '''.strip().replace('\n',' ')
            self.exec_query(q)
        self.commit_with_lock()

    def parse_track_on_href(self, href, tracks):
        '''
        parse tracks from single href
        '''
        cur_set = []
        for t in tracks:
            if t is None: continue
            # remove chars between/including [] and ()
            tmp = re.sub("[\(\[].*?[\)\]]", "", t)
            tmp = tmp.replace('"',"'")
            # remove whitespace
            tmp = tmp.strip()
            # unknown song
            if tmp == "?":
                continue
            # artist/song delimiter
            tmp = tmp.split(' - ')
            if len(tmp) >= 2:
                cur_set.append((tmp[0], tmp[1]))

        cur_set = list(set(cur_set))
        return {'href':href, 'set':cur_set}

    def parse_tracks(self, tracks):
        out = []
        for track in tracks:
            if track is None: continue
            for href,trks in track:
                cur_set = []
                for t in trks:
                    if t is None: continue
                    # remove chars between/including [] and ()
                    tmp = re.sub("[\(\[].*?[\)\]]", "", t)
                    tmp = tmp.replace('"',"'")
                    # remove whitespace
                    tmp = tmp.strip()
                    # unknown song
                    if tmp == "?":
                        continue
                    # artist/song delimiter
                    tmp = tmp.split(' - ')
                    if len(tmp) >= 2:
                        cur_set.append((tmp[0], tmp[1]))

                cur_set = list(set(cur_set))
                out.append({'href':href, 'set':cur_set})

        return out

    def artist_find_artist(self, artist):
        url = f'https://www.mixesdb.com/w/Special:Search?fulltext=Search&cat=Artist&search={artist}'
        soup = self.get_pagex(url)

        lis = soup.find_all('li')
        if lis:
            for li in lis:
                art = li.find('div', {'class':'mw-search-result-heading'})
                if not art:
                    continue

    def artist_get_artist(self, artist, href=None):
        '''
        search for artist. if found, return all mix urls
        from that artist, without getting tracks from mixes
        '''
        
        if href is None:
            df = self.artist_from_artist(artist)
        else:
            df = self.artist_from_href(href)

        if df is None:
            return None

        # get list of mix hrefs belonging to each artist
        href_list = []
        for i,row in df.iterrows():
            url = row[SqlCols.href.value]
            aid = row[SqlCols.prim.value]
             
            # is mix url in mdb_mix_urls ? 
            q = f'''
            select * from {self.mix_urls}
            where {SqlCols.href.value} = "{url}"
            '''.strip().replace('\n',' ')
            # url_df = pd.read_sql(q, self.conn)
            url_df = self.read_sql_with_lock(q)
            # TODO when to check for new mixes
            if url_df.empty:
                soup = self.get_pagex(url)
                # get mix hrefs from artist's page
                hrefs = self.mixes_for_artist(soup)
                urls = []
                for h in hrefs:
                    cur = {}
                    cur[SqlCols.href.value] = url[:200]
                    cur[SqlCols.art_id.value] = aid
                    cur[SqlCols.mix_url.value] = h[:200]
                    urls.append(cur)

                urls_df = None
                if hrefs:
                    # add mixes to `self.mix_urls`
                    urls_df = pd.DataFrame(urls, columns=urls[0].keys())
                    self.to_sql_with_lock(urls_df, self.mix_urls)

                href_list.append((aid, urls_df))

                # add relationships to `self.mdb_artists`
                subs = self.subcategories_for_artist(soup)
                self.add_relationship_to_artist(url, subs)

            else:
                href_list.append((aid, url_df))

        lg = None ; aid = -1
        for f in href_list:
            _aid = f[0] ; _lg = f[1]
            if _lg is None:
                continue
            if lg is None or len(_lg) > len(lg):
                aid = _aid
                lg = _lg

        rels = []
        if lg is not None:
            hr = lg.iloc[0][SqlCols.href.value]  
            rels = self.relations_from_artist_href(hr)

        return (aid,lg,rels)

    def artist_find_correct_artist(self, artist, target_href):
        url = f'https://www.mixesdb.com/w/Special:Search?fulltext=Search&cat=Artist&search={artist}'
        soup = self.get_pagex(url)
        maybes = []
        all_maybes = []
        addl = ['MemberOf', 'AliasOf', 'Category']
        add_artists = []
        lis = soup.find_all('li')
        if lis:
            for li in lis:
                results = li.find('div', {'class':'mw-search-result-heading'})
                if results:
                    for res in results:
                        txt = res.find('a')
                        if txt == -1: continue
                        txt = txt.text
                        if txt.find('Category:') == 0:
                            txt = txt[len('Category:') :]
                        comp_url = self.base_url + res.find('a')['href']
                        all_maybes.append((txt, comp_url))
                        if txt.lower().find(artist.lower()) != 0:
                            continue
                        maybes.append(comp_url)

                    sr = li.find('div',{'class':'searchresult'})
                    if sr:
                        if any(a in sr.text for a in addl) and artist.lower() in sr.text.lower():
                            aname = li.find('a').text
                            aname = aname[len('Category:'):]
                            add_artists.append(aname),

        addl_artists = []
        for aa in add_artists:
            u = [j for i,j in all_maybes if i==aa]
            if u:
                addl_artists.append({'name':aa, 'href':u[0]})

        # if two artists have the same name, grab the more 'popular' one (i.e. has more mixes)
        cnt = 0 ; hrefs = None ; cur_art_href = None ; 
        for art in maybes:
            if target_href and art != target_href:
                continue
            cur_soup = self.get_pagex(art)
            cur_hrefs = self.mixes_for_artist(cur_soup)
            if cnt == 0 or len(hrefs) > cnt:
                hrefs = cur_hrefs
                cur_art_href = art 
                cnt = len(hrefs)

        # return hrefs belonging to desired artist and other artist directly related
        return (hrefs, cur_art_href , addl_artists)

    def subcategories_for_artist(self, soup):
        subs = soup.find('div', {'id':'mw-subcategories'})
        out = []
        if subs is not None:
            for li in subs.find_all('li'):
                cur = {}
                cur[SqlCols.name.value] = li.text
                cur[SqlCols.href.value] = self.base_url + li.find('a')['href']
                out.append(cur)

        links = soup.find('div', {'id':'mw-normal-catlinks'})
        if links is not None:
            for li in links.find_all('li'):
                if li.text == 'Artist':
                    continue
                cur = {}
                cur[SqlCols.name.value] = li.text
                cur[SqlCols.href.value] = self.base_url + li.find('a')['href']
                out.append(cur)

        if out:
            return pd.DataFrame(out, columns=out[0].keys())
        else:
            return None

    def add_relationship_to_artist(self, url, subs):
        if subs is None:
            return 
        ids = []
        for i,sub in subs.iterrows():
            hr = sub[SqlCols.href.value]
            art_df = self.artist_from_href(hr)
            if art_df is not None:
                art_id = art_df[SqlCols.prim.value].iloc[0]
                ids.append(str(art_id))

        ids = list(set(ids))
        rel = '|'.join(ids)
        # only update if NULL
        q = f''' update {self.artist_tbl} 
        set {SqlCols.rel.value} = "{rel}"
        where {SqlCols.href.value} = "{url}"
        and {SqlCols.rel.value} is NULL 
        '''.strip().replace('\n',' ')
        self.exec_query(q)
        self.commit_with_lock()

    def mixes_for_artist(self, soup):
        '''
        Given page of artist, finds all mixes and charts hrefs
        '''
        hrefs = []

        # get charts
        while 1:
            charts = soup.find('div', {'class':'dplCatTable'})
            if not charts:
                break
            lis = charts.find_all('li')
            if not lis:
                break
            for li in lis:
                hrefs.append(self.base_url + li.a['href'])
            break

        # get mixes
        while 1:
            mixes = soup.find('ul', {'id':'catMixesList'})  
            if not mixes:
                break
            for mix in mixes:
                if mix == '\n': continue
                hrefs.append(self.base_url + mix.a['href'])

            nxt_pg = self.nxt_url_from_soup(soup)
            if not nxt_pg:
                break
            soup = self.get_pagex(nxt_pg)

        return hrefs

    def mixes_tracklist_for_mix_href(self, href):
        '''
        Given a href of a mix, return tracklist
        '''
        soup = self.get_pagex(href)
        tracks = []

        divs = soup.find_all('div', {'class':'list-track'})
        if divs:
            for trk in divs:
                tracks.append(trk.text.strip())

        ol = soup.find_all('ol')
        if ol:
            for ols in ol:
                for li in ols.find_all('li'):
                    tracks.append(li.text.strip())

        logger.info(f"Getting {len(tracks)} tracks for {href}")
        return (href,tracks)

    def get_all_artists(self):
        rl = 'https://www.mixesdb.com/w/Category:Artist'

        while 1:
            arts = []
            logger.info(f'{url=}')
            soup = self.get_pagex(url)
            results = soup.find('div',{'id':'mw-subcategories'})
            if results:
                lis = results.find_all('li')
                if lis:
                    for li in lis:
                        cur = {}
                        cur[SqlCols.name.value] = li.text
                        cur[SqlCols.href.value] = self.base_url + li.find('a')['href']
                        arts.append(cur)
        
            if arts:
                df = pd.DataFrame(arts, columns=arts[0].keys())
                # df.to_sql(self.artist_tbl, self.engine, if_exists='append', index=False) 
                self.to_sql_with_lock(df, self.artist_tbl)

            raise Exception('returns tuple change me!!')
            url = self.nxt_url_from_soup(soup)
            if not url:
                break

    def nxt_url_from_soup(self,soup):
        nxt = soup.find('div', {'class':'listPagination'}) 
        if nxt:
            urls = nxt.find_all('a')
            if not urls: 
                return None
            if len(urls) == 2:
                after = urls[1]['href']
                return self.base_url + after
            elif len(urls) == 1:
                if urls[0].text.find('next') == -1:
                    return None
                after = urls[0]['href']
                return self.base_url + after
            else:
                return None
        else:
            return None

                
    
    # TODO
    def update_mixes_hrefs(self):
        '''
        for all artists in mdb_artists
        obtain their current hrefs
        add the hrefs that dont exist in mdb_artists
        to that table
        '''
        pass

    #######################################
    #######################################
    #######################################
    ### BULK PROCESSING ... RUN ON CRON ### 
    #######################################
    #######################################
    #######################################

    def artists_get_all(self):
        '''
        get all artist data
        '''

        def _artists_get_all(tid, adf, mix_df):
            cnt = 0 
            try:
                for i,art in adf.iterrows():
                    cnt += 1
                    logger.info(f"{tid} : {cnt}")
                    # add artist's mix urls to mdb_mix_urls
                    # AND add artist-to-artists relationship in mdb_artists
                    cur_art = art[SqlCols.name.value]
                    logger.info(f"{cur_art}")
                    aga = self.artist_get_artist(cur_art , art[SqlCols.href.value])
                    if aga is None:
                        logger.info(f'MixesDB not found {cur_art}')
                        continue
                    aid,df,rels = aga
                    if aid == -1:
                        logger.info(f'MixesDB artist id not found {cur_art}')
                        continue 
                    if df is not None:
                        hrefs = df[SqlCols.mix_url.value]
                        href_vals = hrefs[~hrefs.isin(mix_df[SqlCols.set_href.value])]
                        for h in href_vals.values:
                            logger.info(f'Receive tracks from {cur_art}')
                            self.receive_tracks_from_href(aid, h)
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                os._exit(1)

        '''
        retrieve all tracks from all artists, updating several db's in process
        '''
        q = f'select * from {self.mixes_tbl}'
        mix_df = self.read_sql_with_lock(q)

        q = f'select * from {self.artist_tbl}'
        art_df = self.read_sql_with_lock(q)

        track_arr = np.array_split(art_df, 10)

        threads = []
        for i, ta in enumerate(track_arr):
            t = threading.Thread(target=_artists_get_all, args=(i, ta, mix_df))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def genres_get_all(self):
        '''
        get all genre information off mixes
        '''
        genres = list(mdb_gen_to_spot_gen.keys())
        for g in genres:
            self.set_genre_data(g)

    def set_genre_data(self, genre):
        '''
        Get tracks related to genre
        '''
        logger.info(f"Finding genre information for {genre}")
        gen_name,gen_count,gen_url = self.genre_find_genre(genre)

        if gen_url is None:
            return None
    
        mixes = self.genre_collect_mixes_from_pg(gen_name, gen_url)
        mixdf = pd.DataFrame(mixes, columns=['mix'])

        # only get tracks from hrefs we dont have
        q = f'select {SqlCols.set_href.value} from {self.mixes_tbl}'
        gen_mixes = pd.read_sql(q, self.conn).drop_duplicates()
        toget = mixdf[~mixdf.mix.isin(gen_mixes[SqlCols.set_href.value])] 
        hrefs = list(toget.mix)

        if hrefs:
            self.receive_track_info(-1 , hrefs)
            # self.db_add_genre_sets(mixes)

    def genres_spotify_lookup(self):
        '''
        after you get all genre mixes/tracks, we want to look all those tracks up
        in spotify and add spotify song_ids to mdb_tracks table 
        '''

        logger.info(f"Reading all tracks from {self.mixes_tbl}")
        tracks = self.read_sql_with_lock(f'select * from {self.mixes_tbl}')
        logger.info(f"Collecting genre information for all genres")
        genres = list(mdb_gen_to_spot_gen.keys())
        gens_df = self.get_genres_data(genres, tracks)

        trks_out = []
        for gd in gens_df:
            logger.info(f"Running {gd['genre']}")
            df = gd['df']
            df = df[df.song_id.isnull()] # only want to check for null song ids
            trks = self.user.set_spotify_id_for_tracks(df, self.mixes_tbl)
            trks_out.append(trks)
