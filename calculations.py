z = 1

def performance(diff, xdiff, dr):
    if(diff > 0):
        return performance_win(diff, xdiff, dr)
    if(diff < 0):
        return performance_lose(diff, xdiff, dr)
    return performance_draw(diff, xdiff, dr)

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

def performance_draw(diff, xdiff, dr):
    factor = factor_draw(diff, xdiff)
    t  = 10 - (dr/10)
    l = factor * diff * (1/pow(10,z)) * pow(t, z)
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

def factor_draw(diff, xdiff):
    if(xdiff > 0):
        return 1 + (xdiff / 10)
    if(xdiff < 0):
        return 1 + (xdiff / 10)
    return 1
