import sqlite3

con = sqlite3.connect(r"C:\Users\rlaxo\OneDrive\바탕 화면\SQLite\kospi.db")
type(con)
# sqlite3.Connection
cursor = con.cursor()
# cursor.execute("CREATE TABLE kakao(Date text, Open int, High int, Low int, Closing int, Volumn int)")
# cursor.execute("INSERT INTO kakao VALUES('16.06.03', 97000, 98600, 96900, 98000, 321405)")
# con.commit()
# con.close()
cursor.execute("SELECT * From kakao")
kakao = cursor.fetchall()
print(kakao)