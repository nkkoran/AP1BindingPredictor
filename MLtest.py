from sklearn import svm
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
import pickle
from sklearn.model_selection import train_test_split



def getSequences():
    f = open("posNeg.pkl", 'rb')

    array = pickle.load(f)

    allMotifs = array[0]
    posMotifs = array[1]
    negMotifs = array[2]
    chromosomes = array[3]

    sequences = {}  # chromosme, sequence

    for chromosme in chromosomes:
        with open(f"/Users/nicholaskoran/Downloads/chromFa/{chromosme}.fa", "r") as inFile:
            for record in SeqIO.parse(inFile, "fasta"):
                sequences[record.id] = record.seq
            inFile.close()

    out = open("sequences.pkl", "wb")
    pickle.dump(sequences, out)
    out.close()

    print(sequences)


def getMotifSequences():
    f = open("posNeg.pkl", 'rb')

    array = pickle.load(f)

    allMotifs = array[0]
    posMotifs = array[1]
    negMotifs = array[2]

    random.shuffle(negMotifs)
    random.shuffle(posMotifs)

    negMotifs = negMotifs[0:len(posMotifs) - 1]

    # Get sequences from pickle
    f = open("sequences.pkl", 'rb')
    sequences = pickle.load(f)

    # count = 0
    # for motif in posMotifs:
    #     if motif[3] == "+\n":
    #         count +=1
    # print("Count",count)

    posSequences = []
    # Iterate through pos cases
    for motif in posMotifs:
        print(len(posSequences), "/", len(posMotifs))
        sequence = sequences[motif[2]]
        if motif[3] == "+\n":
            posSequences.append((str(sequence[int(motif[0]):int(motif[1])])).upper())
        elif motif[3] == "-\n":
            # print("-")
            temp = sequence[int(motif[0]):int(motif[1])]
            temp = str(temp[::-1]).upper()
            out = temp.translate(temp.maketrans('ATCG', 'TAGC'))

            posSequences.append(out)

    print("hello")
    negSequences = []
    # Iterate through pos cases
    for motif in negMotifs:
        print(len(negSequences), "/", len(negMotifs))
        sequence = sequences[motif[2]]
        if motif[3] == "+\n":
            negSequences.append((str(sequence[int(motif[0]):int(motif[1])])).upper())
        elif motif[3] == "-\n":
            temp = sequence[int(motif[0]):int(motif[1])]
            temp = str(temp[::-1]).upper()
            out = temp.translate(temp.maketrans('ATCG', 'TAGC'))
            negSequences.append(out)

    print(posSequences)
    print(negSequences)

    out = open("motifSequences.pkl", "wb")
    pickle.dump([posSequences, negSequences], out)
    out.close()


# getMotifSequences()

f = open('motifSequences.pkl', 'rb')
array = pickle.load(f)

posSequences = array[0]
posSequences = posSequences[0:len(posSequences) - 1]
negSequences = array[1]


def createStartSets(posSequences, negSequences):
    dataSet = []
    labelSet = []
    for sequence in posSequences:
        labelSet.append(1)
        dataSet.append(convertToOneHot(sequence))
    for sequence in negSequences:
        labelSet.append(0)
        dataSet.append(convertToOneHot(sequence))


    print(len(dataSet))
    print(len(labelSet))
    return [dataSet, labelSet]


# start index, end index, chromosome, +/-
def convertToOneHot(sequence):
    # get sequence into an array
    seq_array = list(sequence)

    # integer encode the sequence
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(seq_array)

    # one hot the sequence
    onehot_encoder = OneHotEncoder(sparse=False)
    # reshape
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)

    return onehot_encoded_seq

print(len(posSequences))
print(len(negSequences))
X_train, X_test, y_train, y_test = train_test_split(createStartSets(posSequences, negSequences)[0], createStartSets(posSequences, negSequences)[1], test_size=0.3, random_state=109)

