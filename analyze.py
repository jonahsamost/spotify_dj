import logging
logger = logging.getLogger('Spotify')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

import matplotlib.pyplot as plt
from backend import *

class Analyze():
    def __init__(self):
        super(Analyze, self).__init__()


    def graph_songs(self, df):
        plots = []
        plots.append(('dance', df.danceability,(0,1)))
        plots.append(('energy', df.energy,(0,1)))
        plots.append(('loud', df.loudness,(-80,10)))
        plots.append(('speech', df.speechiness,(0,1)))
        plots.append(('acoust', df.acousticness,(0,1)))
        plots.append(('instru', df.instrumentalness,(0,1)))
        plots.append(('livey', df.liveness,(0,1)))
        plots.append(('valence', df.valence,(0,1)))
        plots.append(('tempo', df.tempo,(20,240)))

        cols = 3
        rows = 3
        positions = cols * rows

        plt.cla() ; plt.clf() ;
        fig = plt.figure(1)
        for i,row in enumerate(plots):
            ax = fig.add_subplot(rows,cols,i+1)
            lr,hr = row[2]
            freq,bins = np.histogram(row[1],bins=50,range=[lr,hr])
            ax.plot(bins[:10],freq[:10])
            ax.set_title(row[0])
        plt.show()


    def set_sid_distances(self, u_df , t_df):
        def euclid_distance(dic, row):
            val = 0
            for k in dic.keys():
                dic_med = dic[k]['med']
                cur_val = row[k]
                val += (dic_med - cur_val)**2
            return val**.5

        cols = [ SqlCols.dance.value, SqlCols.energy.value, SqlCols.loud.value,
            SqlCols.speech.value, SqlCols.acoust.value, SqlCols.instru.value,
            SqlCols.live.value, SqlCols.valen.value, SqlCols.tempo.value]

        u_dic = {}
        for c in cols:
            cur = u_df[c]
            u_dic[c] = {'med':cur.median(), 'std':cur.std()}

        for i,row in t_df.iterrows():
            t_df.loc[i,'distance'] = euclid_distance(u_dic, row)

    def get_sample_sids(self, u_df, t_df):
        umean = u_df['distance'].mean()
        ustd  = u_df['distance'].std()
        sa = umean + ustd ; sb = umean - ustd
        sa2 = umean + 2 * ustd ; sb2 = max(0,umean - 2*ustd)

        std1 = t_df[(t_df['distance'] > sb) & (t_df['distance'] < sa)]
        s1   = std1.sample(n=32)

        std2_a = t_df[(t_df['distance'] > sa) & (t_df['distance'] < sa2)]
        std2_b = t_df[(t_df['distance'] > sb2) & (t_df['distance'] < sb)]
        std2 = std2_a.append(std2_b)
        s2   = std2.sample(13)
        
        std3 = t_df[t_df['distance'] > sa2]
        s3   = std3.sample(5)

        samp = s1.append(s2)
        samp = samp.append(s3)

        return list(samp[SqlCols.song_id.value])
