import pickle
import sys
import os

def getPwm(tf):
    chosen_TF = tf
    threshold = 5

    new_PWN = open(f"{chosen_TF}.motif", "w")

    new_PWN.write(f">ATCG\t{chosen_TF}\t{threshold}\n")

    with open("factorbookMotifPwm.txt","r") as fi:
        id = []
        for ln in fi:
            if chosen_TF in ln and "-ext" not in ln:
                id.append(ln[len(chosen_TF):])
    print(id[0])

    x = id[0].split("\t")

    first = x[2].split(",")
    second = x[3].split(",")
    third = x[4].split(",")
    fourth = x[5].split(",")

    for i in range(len(first)):
        new_PWN.write(f"{first[i]}\t{second[i]}\t{third[i]}\t{fourth[i]}\n")

    new_PWN.close()

    #os.system(f"/Users/nicholaskoran/Downloads/Homer/bin/scanMotifGenomeWide.pl {chosen_TF}.motif hg19 -bed > allMotifs.bed")

def findLargestTF():
    tfs = {}

    with open("factorbookMotifPos.txt", "r") as inFile:
        for line in inFile:
            if line.split("\t")[4] in tfs.keys():
                tfs[line.split("\t")[4]] += 1
            else:
                tfs[line.split("\t")[4]] = 1

    inFile.close()

    maxNum = 0
    maxTf = ""

    for tf in tfs.keys():
        if tfs[tf] >= maxNum:
            maxNum = tfs[tf]
            maxTf = tf

    print(tfs)
    print(maxTf, maxNum)

    return maxTf

def splitPosNeg():
    allMotifs = []
    chromosomes = []

    print("Here1`")
    choiceMotif = ""
    with open("allMotifs.bed", "r") as allMotifsFile:
        for line in allMotifsFile:
            line1 = line.split("\t")
            choiceMotif = line1[3]
            allMotifs.append((line1[1],line1[2],line1[0],line1[5])) #Index 1, index 2, chromosome, +/-
            if line1[0] not in chromosomes:
                chromosomes.append(line1[0])

    allMotifsFile.close()
    print("here2")
    posMotifs = []
    posIndex = {}
    with open("factorbookMotifPos.txt", "r") as allMotifsF:
        for line in allMotifsF:
            line1 = line.split("\t")
            if line1[4] == choiceMotif:
                posMotifs.append((line1[2], line1[3], line1[1],line1[6]))
                posIndex[(line1[2], line1[3], line1[1],line1[6])] = 1

    newPosMotifs = []
    negMotifs = []
    print("here3")
    for motif in allMotifs:
        try:
            if posIndex[motif] == 1:
                newPosMotifs.append(motif)
        except:
            negMotifs.append(motif)

    print(len(allMotifs))
    print(len(newPosMotifs))
    print(len(negMotifs))

    outFile = open("posNeg.pkl", "wb")


    newArr = [allMotifs, newPosMotifs, negMotifs, chromosomes]

    pickle.dump(newArr, outFile)

    outFile.close()

#transcriptionFactor = findLargestTF()
#getPwm(transcriptionFactor)

splitPosNeg()