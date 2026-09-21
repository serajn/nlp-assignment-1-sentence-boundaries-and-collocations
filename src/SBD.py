import sys

train_file = "../data/train/" + sys.argv[1]

X_train = []

def get_L(L_token):
    return L_token[:-1] if L_token.endswith(".") else None

def get_R(R_token):
    return R_token if R_token  != '' else None

def extract_features(L_token, R_token):
    return [
        get_L(L_token),
        get_R(R_token)
    ]

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
        L_token = columns[1]
        R_token = file.readline().split()[1]

        if L_token.endswith("."):
            X_train.append(extract_features(L_token, R_token))
            