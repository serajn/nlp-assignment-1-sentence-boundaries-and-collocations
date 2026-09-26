import string
import math

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

def count_bigrams(infile):
    with open(infile, "r") as file:
        bigrams = {}

        lines = file.readlines()

        for line in lines:
            tokens = line.split()
            
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i+1])

                if (bigram[0] and all(char in string.punctuation for char in bigram[0])
                        or bigram[1] and all(char in string.punctuation for char in bigram[1])):
                    continue

                if bigram in bigrams:
                    bigrams[bigram] += 1
                else:
                    bigrams[bigram] = 1
                

        with open("bigrams.txt", "w") as outfile:
            outfile.write(f"{bigrams}")

    return bigrams

def calculate_N(bigrams):
    count = 0

    for bigram in bigrams:
        count += bigrams[bigram]

    return count

def calculate_chi_squares(unigrams, bigrams):
    chi_squares = {}

    N = calculate_N(bigrams)
    
    for bigram in bigrams:
        observed_value = bigrams[bigram]
        expected_value = (unigrams[bigram[0]] * unigrams[bigram[1]] / N)

        chi_square = pow(observed_value - expected_value, 2) / expected_value

        chi_squares[bigram] = chi_square

    with open("chi_squares.txt", "w") as outfile:
        outfile.write(f"{chi_squares}")

    return chi_squares

def calculate_pmi(unigrams, bigrams):
    pmi_scores = {}

    N = calculate_N(bigrams)

    for bigram in bigrams:
        pmi_score = math.log((bigrams[bigram] * N) / (unigrams[bigram[0]] * unigrams[bigram[1]]), 2)

        pmi_scores[bigram] = pmi_score

    with open("pmi_scores.txt", "w") as outfile:
        outfile.write(f"{pmi_scores}")

    return pmi_scores

def print_top_twenty_chi_squares(chi_squares):
    top_twenty = sorted(chi_squares.items(), key=lambda item: item[1], reverse=True)[:20]

    for score in top_twenty:
        print(f"{score[0]} {round(score[1], 3)}")


unigrams = count_unigrams(infile)
bigrams = count_bigrams(infile)

chi_squares = calculate_chi_squares(unigrams, bigrams)
pmi_scores = calculate_pmi(unigrams, bigrams)

print_top_twenty_chi_squares(chi_squares)
    


