import logging
logger = logging.getLogger('Spotify')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

from data_types import *
import pandas as pd
import pickle

from analyze import Analyze
from spotify import Spotify
from mixesdb import MixesDB 
from reddit import Reddit

class User(Spotify, Analyze):
    def __init__(self, username):
        super(User, self).__init__(username)
        self.genres = None
        self.artists = None
        self.mdb  = MixesDB(self)
        self.redd = Reddit(self)

    def set_spotify_facts(self):
        logger.info("Getting user spotify information")
        self.t_songs_s   = self.top_songs()
        # self.songs_m = self.top_songs(term=Terms.medium_term)
        self.t_artists   = self.top_artists()
        # self.follow    = self.following
        self.likes     = self.liked_songs
        self.psongs    = self.playlist_songs

        # get music from artists
        self.set_artists()

        # get music from genres
        self.set_genres()

        self.set_songs()

    def set_songs(self):
        songs = []
        songs += self.t_songs_s
        songs += self.likes
        songs += self.psongs
        self.user_songs = songs

    def get_user_songs_features(self):
        logger.info("Getting user song's features")
        sids = self.song_ids_from_Items(self.user_songs)
        i = 0 
        while len(sids) < 130:
            if i == len(self.t_artists):
                break
            aid = self.t_artists[i].id
            tracks = u.sp.artist_top_tracks(aid)['tracks']
            cnt = 0
            for track in tracks:
                tid = track['id']
                if tid not in sids:
                    sids.append(track['id'])
                    cnt += 1
                    if cnt == 4:
                        break
            i += 1
        return self.audio_features_for_tracks(sids)

    def set_artists(self):
        if self.artists is not None:
            return self.artists

        arts = []
        arts += [a.name for s in self.t_songs_s for a in s.artists]
        arts += [a.name for a in self.t_artists]
        arts += [a.name for s in self.psongs for a in s.artists]
        arts += [a.name for s in self.likes for a in s.artists]
        # arts += [a.name for a in follow]

        # TODO only want electronic adjacent artists
        # arts = self.filter_artists(arts)

        self.artists = list(set(arts))

    def top_genres(self):
        if self.genres is None:
            return None
        return list(self.genres.sort_values(by='weight')[-30:].index)

    def set_genres(self):
        if self.genres is not None:
            return self.genres

        song_gen   = self.get_songs_genres(self.t_songs_s)
        song_df    = pd.DataFrame.from_dict(song_gen, orient='index')

        artist_gen = self.get_artists_genres(self.t_artists)
        artist_df  = pd.DataFrame.from_dict(artist_gen, orient='index')

        # follow_gen = self.get_following_genres(self.follow)
        # follow_df  = pd.DataFrame.from_dict(follow_gen, orient='index')

        likes_gen  = self.get_liked_songs_genres(self.likes)
        likes_df   = pd.DataFrame.from_dict(likes_gen, orient='index')

        plist_gen  = self.get_playlist_songs_genres(self.psongs)
        plist_df   = pd.DataFrame.from_dict(plist_gen, orient='index')

        genres = pd.concat([song_df, artist_df, likes_df, plist_df])
        gs = {}
        for name,g in genres.groupby(by=genres.index):
            s = g.sum(axis=0)
            gs[name] = {'cnt':s.cnt, 'weight':s.weight}
        self.genres = pd.DataFrame.from_dict(gs, orient='index')

if __name__ == '__main__':

    u = User('jsamost')

    # fd = open('quadro_out.obj', 'rb')
    # quad = pickle.load(fd)
    # fd.close()

    '''
    user_df = u.get_user_songs_features()

    # add all song's we've obtained to db
    a_sids = u.unsplit_song_ids(quad) # add all features 
    u.db_add_spotify_songs(a_sids)

    # get df of songs we want
    o_sids = u.unsplit_song_ids(quad, one=True) # only get 
    mdb_df = u.db_spotify_songs_for_sids(o_sids)

    u.set_sid_distances(user_df, user_df)
    u.set_sid_distances(user_df, mdb_df)
    ssid_samps = u.get_sample_sids(user_df, mdb_df)
    '''

    fd = open('both.obj', 'rb')
    both = pickle.load(fd)
    fd.close()
    from mixesdb_genres import mdb_gen_to_spot_gen

    # u.set_spotify_facts()
    # artists = u.artists
    # genres = u.top_genres()
    # u.mdb.run(arts,gens)

# TODO -- url shortener (some two way hash function) for these long as fuck urls
# TODO do better parsing of songs and artists 
'''
mixesdb
remixrotation
beatport
reddit

whosampled
soundcloud
pandora -- can we somehow get access to its music genome?
bandcamp
mixcloud

tastedive
songdrop.me
SoundCloud
Mixcloud
spotify
pandora
mixing.dj
youtube
resident advisor
mixmag
discogs.com
following labels

-- I follow labels on soundcloud/youtube and junorecords. (there are too many releases on beatport for this to be meaningful for me) I listen to a boatload of mixes on soundcloud/youtube. Check RA's monthly chart list as well as record reviews. Get emails from junorecords and halcyon, grammaphone records and turntable lab as well. Check the mixes section here and on discogs as well as perusing discogs for things in lists, things recommended or other releases by labels I seem to like. I also like checking out redeye's charts as well, but Juno's just too convenient so it gets the majority of my time.

-- The RYM custom chart  - sort by year/genre of your choice (pick esoteric for lesser-known albums)
Bandcamp is good for more obscure stuff. If you follow artists/labels there, you can check your feed for any new releases. A good starting point is going to a page of an album you really like and clicking on a few users who bought that album, chances are they've bought other similar albums you might enjoy.
Also when you're on a user's profile page and you like a lot of their purchases, you can follow them, and then you get notifications whenever they buy a new album.  I've found some good stuff through that.
RYM == https://rateyourmusic.com/customchart

-- For Spotify I find a song I really like, 
    than i type that song + artist into the search bar. 
    For example: Solitary Daze - Maceo Plex. 
    Than go to playlists, all the playlist you’ll 
    see are people making their own playlist 
    around this particular song without renaming the playlist. 
    Some are shit, but sometimes you’ll find someone 
    with the same taste in music as yourself and 
    it’s smooth sailing from there. 
    I trust the music taste of others more than some algorithm.
1001tracklists
youtube

-- I follow producers and labels that 
    consistently deliver songs I like 
    via their mailing list, Bandcamp, Beatport, Spotify/YT

djcity.com clubkillers.com directmusic service.com liveDJservice.com


'''


