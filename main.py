from whosampled import *
from spotify import *
import logging, time, tempfile


def main():
    # logging.basicConfig(level=logging.INFO)
    name = 'The Notorious B.I.G.'
    fd, path = tempfile.mkstemp()
    print('tempfile : {}'.format(path))

    get_all_who_sampled(name,path)
    get_all_spotify(name,path)

if __name__ == '__main__':
    main()
