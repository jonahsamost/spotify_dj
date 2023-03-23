
    def foo(both):

        q = f'select * from {self.spotify_arts}'
        artist_df = pd.read_sql(q, self.conn)

        q = f'select * from {self.spotify_tbl}'
        song_df = pd.read_sql(q, self.conn)

        songids = []
        for i,tr in both.iterrows():
            sname=tr.song ; aname=tr.artist

            df = artist_df[artist_df.artist == aname]
            if df.empty:
                continue

            cursongs = []
            for i,art_row in df.iterrows():
                aid = art_row.artist_id
                songs = song_df[song_df.artist_id == aid]
                if songs.empty:
                    continue

                fsongs = songs[songs.song.str.find(sname) == 0]
                if fsongs.empty:
                    continue

                cursongs += list(fsongs.song_id)

            if cursongs:
                songids.append(cursongs)

