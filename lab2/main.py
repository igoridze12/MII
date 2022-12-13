import numpy
from random import randint as randint
import matplotlib.pyplot as pyplot

try:
    print("Variant 4")
    print("K = ", end = '')
    K = int(input())
    print("N = ", end = '')
    N = int(input())
    if (N < 1):
        raise Exception("N must me larger than 1")

    A = numpy.array([[randint(-10, 10) for j in range(N)] for i in range (N)])
    print(A)

    remainder = int(N / 2)
    large = N - remainder

    if (N % 2 != 0):
        B = A[0: large, 0: large]
        C = A[0: large, remainder:]
        D = A[remainder:, 0: large]
        E = A[remainder:, remainder:]
    else:
        B = A[0: N // 2, 0: N // 2]
        C = A[0: N // 2, N // 2:]
        D = A[N // 2:, 0: N // 2]
        E = A[N // 2:, N // 2:]

    print("B: ")
    print(B)

    print("C: ")
    print(C)

    print("D: ")
    print(D)

    print("E: ")
    print(E)

    zeroesInUnevenColumnsOfE = numpy.count_nonzero(E[:, ::2] == 0)
    print("Zeroes in uneven columns of E: ")
    print(zeroesInUnevenColumnsOfE)

    negativesInEvenRowsOfE = (E[1::2, :] < 0).sum()
    print("Negatives in even rows of E: ")
    print(negativesInEvenRowsOfE)

    F = A.copy()

    if (zeroesInUnevenColumnsOfE > negativesInEvenRowsOfE):
        print("More zeroes in uneven columns of E, symmetrically switching C and B")
        if (N % 2 != 0):
            C_temp = numpy.flip(B, axis = 0)
            E_temp = numpy.flip(C, axis = 0)
            F[remainder:, remainder:] = B_temp
            F[0: large, remainder:] = C_temp
        else:
            C = numpy.flipud(A[N // 2:, N // 2:])
            E = numpy.flipud(A[0: N // 2, N // 2:])
    else:
        print("More negatives in even rows of E, non-symmetrically switching B and E")
        if (N % 2 != 0):
            F[0: large, 0: large] = B
            F[0: large, remainder:] = E
        else:
            C = A[0: N // 2, 0: N // 2]
            B = A[0: N // 2, N // 2:]

    print("F: ")
    print(F)

    determinatorF = numpy.linalg.det(A)
    print("Determinator of A: ", end = '')
    print(determinatorF)

    diagonalSumF = numpy.trace(F)
    print("Diagonal sum of F: ", end = '')
    print(diagonalSumF)

    if determinatorF > diagonalSumF:
        AT = numpy.transpose(A)
        Ainv = numpy.linalg.inv(A)
        result = numpy.subtract(numpy.multiply(Ainv, AT), numpy.multiply(K, F))
    else:
        AT = numpy.transpose(A)
        G = numpy.tril(A)
        Finv = numpy.linalg.inv(F)
        result = numpy.multiply(numpy.subtract(numpy.add(AT, G), Finv), K)
    print("Result: ", end = '')
    print(result)

    graph1=pyplot.figure()
    eSize=len(E)*len(E[0])
    X = numpy.linspace(0, eSize, eSize)
    Y = numpy.asarray(E).reshape(-1)
    pyplot.plot(X, Y)
    pyplot.title("B")
    pyplot.show()

    graph2 = pyplot.figure()
    bSize = len(B) * len(B[0])
    X = numpy.linspace(0, bSize, bSize)
    Y = numpy.asarray(B).reshape(-1)
    pyplot.plot(X, Y)
    pyplot.title("C")
    pyplot.show()

    graph3 = pyplot.figure()
    dSize = len(D) * len(D[0])
    X = numpy.linspace(0, dSize, dSize)
    Y = numpy.asarray(D).reshape(-1)
    pyplot.plot(X, Y)
    pyplot.title("D")
    pyplot.show()

    graph4 = pyplot.figure()
    cSize = len(C) * len(C[0])
    X = numpy.linspace(0, cSize, cSize)
    Y = numpy.asarray(C).reshape(-1)
    pyplot.plot(X, Y)
    pyplot.title("E")
    pyplot.show()

except ValueError:
    print("Not a number")
