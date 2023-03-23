from enum import Enum

class Terms(Enum):
    short_term='short_term'
    medium_term='medium_term'
    long_term='long_term'

class Item:
    def __init__(self, dic):
        if 'name' in dic:
            self.name = dic['name']
        if 'id' in dic:
            self.id   = dic['id']
        if 'type' in dic:
            self.type = dic['type']
        if 'uri' in dic:
            self.uri = dic['uri']
        if 'popularity' in dic:
            self.popularity = dic['popularity']
        if 'genres' in dic:
            self.genres = dic['genres']
        if 'user_popularity' in dic:
            self.user_popularity = dic['user_popularity']
        if 'artists' in dic:
            self.artists = [Item(art) for art in dic['artists']]

        self.date = None


