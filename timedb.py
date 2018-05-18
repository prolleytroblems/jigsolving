import sqlite3

class DBConnector:

    def __init__(self):
        self.conn=sqlite3.connect("data\puzzletimes.db")
        self.cur=self.conn.cursor()

    def create_table(self):
        self.cur.execute("CREATE TABLE IF NOT EXISTS times (pooling INTEGER, pieces INTEGER, runtime REAL, preptime REAL)")
        self.conn.commit()

    def insert(self, values):
        self.cur.execute("INSERT INTO times VALUES(?, ?, ?, ?) ", (values["pooling"], values["pieces"], values["runtime"], values["preptime"]))
        self.conn.commit()

    def view_all(self):
        self.cur.execute("SELECT * FROM times")
        rows=self.cur.fetchall()
        return(rows)

    def view(self, item):
        self.cur.execute("SELECT * FROM times WHERE pooling=? AND pieces=?", (item["pooling"], item["pieces"]))
        rows=self.cur.fetchall()
        return(rows)

    def delete(self, item):
        self.cur.execute("DELETE FROM times WHERE pooling=? AND pieces=?", (item["pooling"], item["pieces"]))
        self.conn.commit()

    def update(self, values):
        self.cur.execute("UPDATE times SET runtime=? , preptime=? WHERE pooling=? AND pieces=?", (values["times"], values["runtime"], values["pooling"], values["pieces"]))
        self.conn.commit()

    def __del__(self):
        self.conn.close()

if __name__ == '__main__':
    conn=DBConnector()
    #conn.insert({"pooling":4, "pieces":4, "runtime":0.5, "preptime":5.6})
    #conn.delete({"pooling":4, "pieces":4, "runtime":0.5, "preptime":5.6})
    print(conn.view_all())
