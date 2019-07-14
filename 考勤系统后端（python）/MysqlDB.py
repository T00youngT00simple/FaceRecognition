from pymysql import connect
class MyDB(object):
    def __init__(self,host,user,pw,database):
        self.conn = connect(host,user,pw,database)
        self.cs = self.conn.cursor()

    def __enter__(self):
        return self.cs

    def __exit__(self,exc_type,exc_val,exc_tb):
        self.conn.commit()
        self.cs.close()
        self.conn.close()
