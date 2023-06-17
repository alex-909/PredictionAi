import numpy as np
from calculations import * 

def edit_data_set(path):
    data = np.genfromtxt(path, delimiter=',')
    #print(data)
    i = 0 
    # th,lh1,lh2,lh3,lh4,lh5,ta,la1,la2,la3,la4,la5,diff
    arr = []
    for row in data:
        i += 1
        if(i==1): 
            continue
        LH = []
        LA = []
        if(row[15]==0):
            continue
        for i in [1, 4, 7, 10, 13]:
            lh = performance(diff=row[i], xdiff=row[i+1], dr=row[i+2])
            LH.append(lh)
        for i in [ 17, 20, 23, 26, 29]:
            la = performance(diff=row[i], xdiff=row[i+1], dr=row[i+2])
            LA.append(la)
        new_row = []
        new_row.append(19 - row[0])
        new_row += LH
        new_row.append(19 - row[16])
        new_row += LA
        new_row.append(row[32])
        print(new_row)
        arr.append(new_row)
    np.set_printoptions(suppress=True)
    a = np.asarray(arr)
    np.savetxt("data\\train_data_edit.csv", a, delimiter=",", fmt='%5.2f')


path = "data\\train_data_bl.csv"
edit_data_set(path)