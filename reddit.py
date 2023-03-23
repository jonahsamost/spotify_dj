import praw
import re
from db import MyDB
from backend import Backend, SqlCols
from enum import Enum
from reddit_genres import rgs

'''
this is interesting
https://old.reddit.com/r/electronicmusic/comments/72l7ww/extremely_genre_specific_relectronicmusic/

list of all music subreddits
https://old.reddit.com/r/Music/wiki/musicsubreddits
'''


class RedditTypes(Enum):
    HOT='hot'
    NEW='new'
    RISING='rising'
    CONTRO='controversial'
    TOP='top'

class Reddit(MyDB):
    def __init__(self, user):
        super(Reddit, self).__init__()

        self.client_id = ''
        self.secret    = ''
        self.redirect  = 'http://jasvandy.github.io'
        self.red = praw.Reddit(client_id=self.client_id, client_secret=self.secret, user_agent='hello')

        self.redd_subrs =  'redd_subs'
        self.redd_tracks = 'redd_tracks'

        self.user = user

        self.db_create_tables()

    def db_create_tables(self):
        q = f'''
        create table if not exists `{self.redd_subrs}` (
        {SqlCols.prim.value} INT NOT NULL AUTO_INCREMENT,
        {SqlCols.name.value} varchar(80) character set utf16,
        PRIMARY KEY ({SqlCols.prim.value})
        ) ENGINE = InnoDB
        '''.strip().replace('\n','')
        self.exec_query(q)

        q = f'''
        create table if not exists {self.redd_tracks} (
        {SqlCols.subr.value} INT NOT NULL default -1 ,
        {SqlCols.artist.value} varchar(100) character set utf16,
        {SqlCols.song.value} varchar(100) character set utf16
        ) ENGINE = InnoDB
        '''.strip().replace('\n','')
        self.exec_query(q)

        self.conn.commit()

    def genres_get_genres(self, genres):
        gens = []
        for g in genres:
            for k,v in rgs.items():
                if g in v:
                    gens.append(k)
        self.genres = list(set(gens))

    def parse_tracks(self, tracks):
        out = []
        for t in tracks:
            if t is None: continue
            # remove chars between/including [] and ()
            tmp = re.sub("[\(\[].*?[\)\]]", "", t)
            # remove whitespace
            tmp = tmp.strip()
            # artist/song delimiter
            tmp = tmp.split(' - ')
            if len(tmp) >= 2:
                out.append((tmp[0], tmp[1]))

        return list(set(out))

    def get_subreddit_tracks(self, subr, which='hot', limit=200):
        
        sub = self.red.subreddit(subr)
        if which==RedditTypes.HOT.value:
            sub = sub.hot(limit=limit)
        elif which==RedditTypes.NEW.value:
            sub = sub.new(limit=limit)
        elif which==RedditTypes.RISING.value:
            sub = sub.rising(limit=limit)
        elif which==RedditTypes.CONTRO.value:
            sub = sub.controversial(limit=limit)
        elif which==RedditTypes.TOP.value:
            sub = sub.top(limit=limit)
        else:
            assert False,'Incorrect subreddit type'

        tracks = [track.title for track in sub]
        return self.parse_tracks(tracks)

