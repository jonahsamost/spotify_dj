import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.dialects import mysql as mysql_types
from datetime import date, timedelta
import pandas as pd
import numpy as np
import math
# from multiprocessing import Lock
from threading import Lock

import logging
logger = logging.getLogger('Spotify')

class MyDB(object):
    def __init__(self):
        super(MyDB,self).__init__()
        self.isSet = False
        self.lastError = ''
        self.db_lock = Lock()
        try:
            self.u = 'spotify'
            self.p = 'password'
            self.host = '127.0.0.1'
            self.db_name = 'music'
            self.conn = mysql.connector.connect(user=self.u,password=self.p,host=self.host,database=self.db_name)
            self.curs = self.conn.cursor()
            self.eng = create_engine(f'mysql+pymysql://{self.u}:{self.p}@{self.host}/{self.db_name}')
            self.isSet = True
        except Exception as e:
            self.lastError = e
            return
    
    @property
    def connection(self):
        return self.conn
    @property
    def cursor(self):
        return self.curs
    @property
    def engine(self):
        return self.eng

    def terminate(self):
        self.connection.disconnect()
        self.cursor.close()

    def exec_query(self,query,multi=False):
        try:
            self.db_lock.acquire()
            if not multi:
                self.cursor.execute(query)
                r = self.cursor.fetchall()
                self.lastError = 0 
                return r
            else:
                for res in self.cursor.execute(query,multi=True):
                    r = self.cursor.fetchall()
                self.lastError = 0 
                return r1
        except Exception as e:
            self.lastError = e 
            return False
        finally:
            self.conn.commit()
            self.db_lock.release()


    def table_exists(self,table):
        query = f"show tables like '{table}'"
        return self.exec_query(query)

    def read_sql_with_lock(self, query):
        df = None
        try:
            self.db_lock.acquire()
            df = pd.read_sql(query, self.conn)
        except:
            return None
        finally:
            self.db_lock.release()

        return df
    
    def to_sql_with_lock(self, df, name): 
        try:
            self.db_lock.acquire()
            df.to_sql(name, self.engine, if_exists='append', index=False) 
            self.conn.commit()
        finally:
            self.db_lock.release()

    def commit_with_lock(self):
        try:
            self.db_lock.acquire()
            self.conn.commit()
        finally:
            self.db_lock.release()

