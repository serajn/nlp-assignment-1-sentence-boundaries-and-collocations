import sys

train_file = "../data/train/" + sys.argv[1]

with open(train_file, "r") as file:
    L_counts = {}

    for line in file:
        columns = line.split()
        token_number = columns[0]
        token = columns[1]
        label = columns[2]

        if token.endswith("."):
            L = token[:-1]

            if L in L_counts:
                L_counts[L] += 1
            else:
                L_counts[L] = 1

            ##count = L_counts[L]

            ##print(token_number, token, L, label)
    
    file.seek(0)

    for line in file:
        columns = line.split()
        token = columns[1]
        label = columns[2]

        if token.endswith("."):
            L = token[:-1]

            if L in L_counts:
                count = L_counts[L]
                print(L, count, label)