import string

infile = "../data/train/Collocations"

def count_unigrams(infile):
    with open(infile, "r") as file:
        unigrams = {}

        lines = file.readlines()

        for line in lines:
            tokens = line.split()

            for token in tokens:
                if token and all(char in string.punctuation for char in token):
                    continue

                if token in unigrams:
                    unigrams[token] += 1
                else:
                    unigrams[token] = 1

        with open("unigrams.txt", "w") as outfile:
            outfile.write(f" {unigrams}")
    return unigrams       
