def addToList(a, b):
    if not type(a) == type([]):
        a = [a]
    if not type(b) == type([]):
        b = [b]
    return a + b


def crossProdAux(A, B):
    return [addToList(a, b) for a in A for b in B]


def crossProduct(sup):
    prod = list([addToList(a, []) for a in sup[0]])
    for set in sup[1:]:
        prod = crossProdAux(prod, set)

    return prod
