import psycopg2
import pandas as pd 


def get_data():
    df = pd.read_csv('../dataset/train.csv')
    rename_rule = {col: f'f{i}' for i, col in enumerate(df.columns[2:])}
    df = df.rename(columns=rename_rule)
    return df

def insert_data(db_connection, data):
    insert_query = '''
    INSERT INTO train 
        (id, target, ''' + ', '.join([f'f{i}' for i in range(300)]) + ') ' + \
        'VALUES (' + ', '.join(['%s' for _ in range(302)]) + ');'
    print(insert_query)
    with db_connection.cursor() as cur:
        for i, row in data.iterrows():
            cur.execute(insert_query, row)
        db_connection.commit()


if __name__ == '__main__':
    db_connect = psycopg2.connect(
        user='username',
        password='password',
        host='localhost',
        port=5432,
        database='database'
    )
    df = get_data()
    insert_data(db_connect, df)
    