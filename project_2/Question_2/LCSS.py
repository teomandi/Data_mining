import sys
sys.path.insert(0, '..')
from HarversineDistance import harversine

def LCS(X, Y):
    m = len(X)
    n = len(Y)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]: 
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    print(FindCommonPath(C,Y))
    return C

def LonsLatsLCS(X, Y):
    m = len(X)
    n = len(Y)
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            harv = harversine(X[i-1][1],X[i-1][2],Y[j-1][1],Y[j-1][2])
            #print "harv = ", harv
            if( harv <= 0.2):
                #print "!S+O+S!", harv
                C[i][j] = C[i-1][j-1] + 1

            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    commonPath = FindCommonPath(C, Y)
    returnList = [max(max(C)), commonPath]
    return returnList

def FindCommonPath(C,Y):
    commonPath = []
    i=len(C)-1
    j=len(C[len(C)-1])-1
    while(True):
        if(j == 0 and i == 0): #break case
            break;
        if(j!=0 and C[i][j] == C[i][j-1]) :
            j=j-1
            continue
        elif(i!=0 and C[i][j] == C[i-1][j]):
            i=i-1
            continue
        else:
            #print Y[j-1]
            commonPath.insert(0, Y[j-1]) #j because its the word of columns
            i=i-1
            j=j-1
    return commonPath