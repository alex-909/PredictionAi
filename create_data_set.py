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

def insert_csv(rows):
    #diff, xdiff, tdiff x10
    with open('data\\train_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = [
            "th", # 1 in

            "diffh1", "xdiffh1", "tdiffh1", # 15 in
            "diffh2", "xdiffh2", "tdiffh2",
            "diffh3", "xdiffh3", "tdiffh3",
            "diffh4", "xdiffh4", "tdiffh4",
            "diffh5", "xdiffh5", "tdiffh5",

            "ta", # 1 in

            "diffa1", "xdiffa1", "tdiffa1", # 15 in
            "diffa2", "xdiffa2", "tdiffa2",
            "diffa3", "xdiffa3", "tdiffa3",
            "diffa4", "xdiffa4", "tdiffa4",
            "diffa5", "xdiffa5", "tdiffa5",

            "hs", #2 out
            "as"
            ]
        # => 32 in
        # => 2  out 

        writer.writerow(field)
        #print(rows)
        for row in rows:
            writer.writerow(row)

def get_xg(s, sot, c): #shots, shotsontarget, woodwork, corners
    s = int(s)
    sot = int(sot)
    c = int(c)

    xg = 0.5*sot + 0.3*c + 0.2*(s-sot)
    return xg

def get_last_five_games(game):

    home_team = game['HomeTeam']
    away_team = game['AwayTeam']

    saison = game['Saison']
    spieltag = int(game['Spieltag'])

    homeGames = []
    awayGames = []
    for i in [1,2,3,4,5]:
        q = query(f'SELECT * FROM Bundesliga WHERE Saison = "{saison}" AND Spieltag = {str(spieltag-i)} AND (HomeTeam="{home_team}" OR AwayTeam = "{home_team}")')
        homeGames.append(json.loads(q.text)[0])

        q = query(f'SELECT * FROM Bundesliga WHERE Saison = "{saison}" AND Spieltag = {str(spieltag-i)} AND (HomeTeam="{away_team}" OR AwayTeam = "{away_team}")')
        awayGames.append(json.loads(q.text)[0])

    return homeGames, awayGames

def create_row(game, homeGames, awayGames):
    home_team = game['HomeTeam']
    away_team = game['AwayTeam']
    
    th = game['GTPH']
    ta = game['GTPA']

    #hometeam data
    row = []
    row.append(th)
    for i in range(5):
        hg = homeGames[i]

        # diffh
        diffh = int(hg['FTHG']) - int(hg['FTAG'])
        

        #xdiffh
        xdiffh = float(get_xg(s=hg['HS'], sot=hg['HST'], c=hg['HC'])) - float(get_xg(s=hg['AS'], sot=hg['AST'], c=hg['AC']))

        #tdiffh
        tdiffh = int(hg['GTPA']) - int(hg['GTPH'])

        if hg['HomeTeam'] != home_team:
            diffh   = -diffh
            xdiffh  = -xdiffh
            tdiffh  = -tdiffh

        row += [diffh, xdiffh, tdiffh]
    
    #awayteam data
    row.append(ta)
    for i in range(5):
        ag = awayGames[i]

        # diffh
        diffa = int(ag['FTHG']) - int(ag['FTAG'])
        

        #xdiffh
        xdiffa = float(get_xg(s=ag['HS'], sot=ag['HST'], c=ag['HC'])) - float(get_xg(s=ag['AS'], sot=ag['AST'], c=ag['AC']))

        #tdiffh
        tdiffa = int(ag['GTPA']) - int(ag['GTPH'])

        if ag['AwayTeam'] == away_team:
            diffa   = -diffa
            xdiffa  = -xdiffa
            tdiffa  = -tdiffa

        row += [diffa, xdiffa, tdiffa]

    row.append(game['FTHG']) #hs homescore
    row.append(game['FTAG']) #as awayscore

    return row

#main start
q = query('SELECT * FROM Bundesliga WHERE Spieltag')
data = json.loads(q.text)

rows = []
i = 0
for game in data:
    print(i)
    i = i+1
    if int(game['Spieltag']) <=5:
        continue

    homeGames, awayGames = get_last_five_games(game)
    row = create_row(game=game, homeGames=homeGames, awayGames=awayGames)
    rows.append(row)

    #if int(game['Spieltag']) > 10:
    #    break

insert_csv(rows)