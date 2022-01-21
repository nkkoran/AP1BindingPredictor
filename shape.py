import pickle

def writeOut():
    f = open('motifSequences.pkl', 'rb')
    array = pickle.load(f)

    posSequences = array[0]
    posSequences = posSequences[0:len(posSequences) - 1]
    negSequences = array[1]

    outFile = open("allMotifs.txt", "w")

    for motif in posSequences:
        outFile.write(f">pos\n{motif}\n")

    for motif in negSequences:
        outFile.write(f">neg\n{motif}\n")

    outFile.close()

def getAttributes():
    outFile = open("shapeArrays.pkl", "wb")
    outArr = []
    for shapeType in ["EP", "HelT", "MGW", "ProT", "Roll"]:
        out = []
        EP = open(f"allMotifs.fa.{shapeType}", "r")
        for line in EP:
            if ">" in line:
                continue
            else:
                a = line.replace("NA", "0")
                out.append([float(x) for x in a.split(",")])
        EP.close()
        outArr.append(out)

    pickle.dump(outArr,outFile)




getAttributes()