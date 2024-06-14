from argparse import ArgumentParser

import pandas as pd 
import psycopg2
import time 

### Overview
# 아래 스크립트를 모아 작성한 스크립트입니다.
# - data_insertion_loop.py
# - data_insertion.py


def get_data():
    df = pd.read_csv('data/train.csv')
    rename_rule = {col: f'f{i}' for i, col in enumerate(df.columns[2:])}
    df = df.rename(columns=rename_rule)
    return df

def create_table(db_connect):
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS train (
        id int PRIMARY KEY NOT NULL,
        target float8 NOT NULL,''' + ', '.join([f'f{i} float8 NOT NULL' for i in range(300)]) \
    + ');'
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()

def insert_data(db_connection, data):
    insert_query = '''
    INSERT INTO train 
        (id, target, ''' + ', '.join([f'f{i}' for i in range(300)]) + ') ' + \
        'VALUES (' + ', '.join(['%s' for _ in range(302)]) + ');'
    print(insert_query)
    with db_connection.cursor() as cur:
        for i, row in data.iterrows():
            cur.execute(insert_query, row)
            time.sleep(1)
            db_connection.commit()
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default='localhost') # args.db_host로 접근할 수 있습니다.
    parser.add_argument("--user", dest="user", type=str)
    parser.add_argument("--password", dest="password", type=str)
    parser.add_argument("--database", dest="database", type=str)
    args = parser.parse_args()
    
    db_connect = psycopg2.connect(
        user=args.user,
        password=args.password,
        host=args.db_host,
        port=5432,
        database=args.database
    )
    create_table(db_connect)
    df = get_data()
    insert_data(db_connect, df)