z = 1

def performance(diff, xdiff, dr):
    if(diff > 0):
        L = performance_win(diff, xdiff, dr)
    elif(diff < 0):
        L = performance_lose(diff, xdiff, dr)
    else:
        L = performance_draw(xdiff, dr)
    
    L = float(f'{L:.2f}')
    return L

def performance_win(diff, xdiff, dr):
    factor = factor_win(diff, xdiff)
    t  = 10 - (dr/10)
    l = factor * diff * (1/pow(10,z)) * pow(t, z)
    return l

def performance_lose(diff, xdiff, dr):
    factor = factor_lose(diff, xdiff)
    t  = 10 + (dr/10)
    l = factor * diff * (1/pow(10,z)) * pow(t, z)
    return l

def performance_draw(xdiff, dr):
    factor = factor_draw(xdiff)
    t  = 10 - (dr/10)
    l = factor * (1/pow(10,z)) * pow(t, z)
    return l

def factor_win(diff, xdiff):
    if(xdiff > 0):
        return xdiff / diff
    if(xdiff < 0):
        return 1 / (1+abs(xdiff))
    return 1 - (diff/10)

def factor_lose(diff, xdiff):
    if(xdiff > 0):
        return 1 / (1 + xdiff)
    if(xdiff < 0):
        return xdiff / diff
    return 1 + (diff / 10)

def factor_draw(xdiff):
    if(xdiff > 0):
        return 1 + (xdiff / 10)
    if(xdiff < 0):
        return 1 + (xdiff / 10)
    return 1
