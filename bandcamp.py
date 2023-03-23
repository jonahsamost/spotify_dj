from bs4 import BeautifulSoup as bs
import requests
import threading
import math

album='album'
track='track'

def get_artist_song_dict():
    f = open('/home/lodi/Downloads/band', 'r')
    soup = bs(f.read(), 'html.parser')
    f.close()

    for o in soup.find_all('ol'):
        try:
            if 'collection-grid' in o['class']:
                break
        except:
            continue


    hits = []
    for li in o.find_all('li'):
        dic = {}
        artist = song = None
        try:
            dtype = li['data-itemtype']
            cur = li.find('div', {'class':'collection-title-details'})
            cur = cur.find('a',{'class':'item-link'})
            href = cur['href']
            for d in cur.find_all('div'):
                if 'collection-item-title' in d['class']:
                    song = d.text.split('\n')[0]
                if 'collection-item-artist' in d['class']:
                    txt = d.text
                    if txt.find('by ') == 0:
                        txt = txt[3:]
                    artist = txt.split('\n')[0]

            dic['artist'] = artist
            dic['song']   = song
            dic['href']   = href
            dic['type']   = dtype
            hits.append(dic)

        except:
            continue 

    return hits

def get_album_hits(hit):
    hits = []
    r = requests.get(hit['href'])
    if r.status_code != 200:
        return None

    soup = bs(r.text, 'html.parser')
    tracks = soup.find_all('table', {'class':'track_list'}) 
    for ctrack in tracks:
        trs = ctrack.find_all('tr', {'class':'track_row_view'})
        for tr in trs:
            title = tr.find('div', {'class':'title'}) 
            title = title.span.text
            if title.find(' - ') != -1:
                try: song, artist = title.split(' - ') 
                except: continue 
            else:
                song = title
                artist = hit['artist']
            cur = {}
            cur['song'] = song
            cur['artist'] = artist
            cur['type'] = track
            cur['href'] = None
            hits.append(cur)

    return hits

def split_arr(arr, size):
    stride = math.ceil(len(arr) / size)
    arrs = [arr[i*stride:i*stride+stride] for i in range(size)]
    return arrs

thrds = 14
album_thread_res = [None] * thrds
def album_track_thread(i, album_list):
    out = []
    for hit in album_list:
        out.append(get_album_hits(hit))

    album_thread_res[i] = out

def get_tracks():
    hits = get_artist_song_dict()
    tracks = []
    albums = [x for x in hits if x['type'] == 'album'] 
    albums = split_arr(albums, thrds)
    threads = []
    for i,albs in enumerate(albums):
        t = threading.Thread(target=album_track_thread, args=(i,albs))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    tracks = []
    for thrd in album_thread_res: 
        for alb in thrd: 
            if alb: 
                tracks += alb 

    for hit in hits:
        if hit['type'] == track:
            tracks.append(hit)

    return tracks


tracks = get_tracks()
