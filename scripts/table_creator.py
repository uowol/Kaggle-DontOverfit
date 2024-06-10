import psycopg2


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
        
        
if __name__ == '__main__':
    db_connect = psycopg2.connect(
        user='username',
        password='password',
        host='localhost',
        port=5432,
        database='database'
    )
    
    create_table(db_connect)
