from thewire import *
import logging, threading, pickle,configparser

def get_all_who_sampled(name,path):

    config = configparser.ConfigParser()
    config.read('../src/config.txt') # TODO 
    threading = False 
    try: threading = int(config['DEFAULT']['THREADING'])
    except: pass
    
    ws = WhoSampled()
    artist_url = ws.search_for_artist(name)
    if artist_url == None:
        return 
    logging.info("artist url : {}".format(artist_url))
    sample_pages = ws.get_tracks_sampled(artist_url)
    logging.info("sample pages: {}".format(sample_pages))
    
    threads = []
    if threading:
        for i in range(1,sample_pages+1):
            t = threading.Thread(target=ws.get_track_connections_per_page, args=(artist_url,i))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    else: 
        print("Getting WhoSampled pages")
        for i in range(1,sample_pages+1):
            ws.get_track_connections_per_page(artist_url,i)
             
    flat_list = [item for sublist in ws.results for item in sublist]
    dedup = [dict(t) for t in {tuple(d.items()) for d in flat_list}]

    with open(path,'wb') as f:
        pickle.dump(dedup,f) 
        print("samples written to temp file")

class WhoSampled():
    def __init__(self):
        self.base_url = 'https://www.whosampled.com'
        self.search_url = '/search/?q='
        self.samples = '/samples'
        self.sp = '/?sp='
        self.results = []

    def go(self, name=None):
        pass

    '''
    Given an artist name, return the artist url if it exists
    '''
    def search_for_artist(self,name):
        search_term = self.base_url + self.search_url + name.replace(' ','+')
        pg = get_page(search_term)
        if not pg: return None
        soup = get_html(pg)
        if not soup: return None
        tophit = get_bs_class(soup,'div','topHit')  
        if not tophit: return None
        if len(tophit) > 1: 
            logging.info("{}: len(tophits) > 1 for url: {}".format(inspect.stack()[0][3] , searchterm))
        artistname = get_bs_class(tophit[0],'a','artistName')
        if not artistname: return None
        suburl = artistname[0].get('href')
        if not suburl: return None
        if suburl[-1] == '/': suburl = suburl[:-1]
        return self.base_url + suburl

    '''
    logs how many samples 
    returns amount of pages of samples
    '''
    def get_tracks_sampled(self,url):
        pg = get_page(url + self.samples)
        if not pg: return None
        soup = get_html(pg)
        if not soup: return None
        sample_found = get_bs_class(soup,'span','section-header-title')
        if not sample_found: return None
        txt = sample_found[0].getText()
        logging.info('{}'.format(txt)) 
        pgcount = get_bs_class(soup,'span','page')
        count = 0
        if not pgcount:
            count = 1
        else:
            count = int(pgcount[-1].getText())
        self.results = [None]*count
        return count 
        

    '''
    given an artist's url (ex. https://www.whosampled.com/Kanye-West/)
    and a page number, return all samples on that page
    '''
    def get_track_connections_per_page(self,artist_url,pagenum):
        url = artist_url + self.samples + self.sp + str(pagenum)
        pg = get_page(url)
        if not pg: return None
        soup = get_html(pg)
        if not soup: return None
        conns = get_bs_class(soup,'div','trackConnections')
        if not conns: return None

        allsamples = []
        for conn in conns:
            # first check for more than 3 samples link
            moresamp = get_bs_class(conn,'a','moreLink bordered-list moreConnections')
            samples = []
            if moresamp:
                moresamp_url = moresamp[0].get('href')
                if moresamp_url:
                    samples = self.get_song_samples_overflow(self.base_url + moresamp_url)
            else: # 3 or less samples exist in this connection
                samples = self.get_song_samples_normal(conn)

            allsamples += samples

        # return allsamples
        self.results[pagenum - 1] = allsamples

    '''
    if 3 or less samples in connection on sample page
    '''
    def get_song_samples_normal(self,soup):
        lis = get_bs_tag(soup,'li') 
        if not lis: return []
        samples = []
        for li in lis:
            allas = li.findAll('a')
            if not allas or len(allas) != 2: continue 
            song = allas[0].getText()
            artist  = allas[1].getText()
            year = search_year(li.getText())
            samples.append(create_song(song,artist,year))
        return samples


    '''
    if 4 or more samples in connection on sample page
    '''
    def get_song_samples_overflow(self,page):
        pg = get_page(page)
        if not pg: return None
        soup = get_html(pg)
        if not soup: return None
        entries = get_bs_class(soup,'div','listEntry sampleEntry')
        samples = []
        for entry in entries:
            song = get_bs_class(entry,'a','trackName playIcon')
            artist = get_bs_class(entry,'span','trackArtist')
            if not song or not artist: continue
            song = song[0].getText()
            art = artist[0].find('a').getText()
            year = search_year(artist[0].getText())
            samples.append(create_song(song,art,year))
        return samples
            
