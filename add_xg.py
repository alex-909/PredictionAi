import pymysql
import requests
import urllib.parse
import json
import csv

url = "https://proxy.cnp-predictions.de/query2.php?sql="

def query(sql_order):
    sql_order = urllib.parse.quote(sql_order)
    #print(url + sql_order)
    r = requests.get(url + sql_order)
    return r

def get_xg(s, sot, c): #shots, shotsontarget, woodwork, corners
    s = int(s)
    sot = int(sot)
    c = int(c)

    xg = 0.5*sot + 0.25*c + 0.25*(s-sot)
    return xg

#main start
q = query('SELECT * FROM Bundesliga')
data = json.loads(q.text)

for game in data:
    ID = game["ID"]
    print(ID)
    xgh = float(get_xg(s=game['HS'], sot=game['HST'], c=game['HC']))
    xga = float(get_xg(s=game['AS'], sot=game['AST'], c=game['AC']))
    diff = xgh - xga
    s = f"UPDATE Bundesliga SET xDiff = {diff} WHERE ID = '{ID}'"
    query(s)

