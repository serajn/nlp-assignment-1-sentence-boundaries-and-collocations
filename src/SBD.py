import sys

train_file = "../data/train/" + sys.argv[1]

X_train = []
Y_train = []

'''
Functions to extract features from the tokens. The features are:
1. L: The token to the left of the period (without the period).
2. R: The token to the right of the period.
3. is_L_less_than_four: A binary feature indicating if the left token is less than four characters long.
4. is_L_a_number: A binary feature indicating if the left token is a number.
5. is_L_capitalized: A binary feature indicating if the left token is capitalized.
6. count_L: A feature indicating the count of how many times the left token appears to the left of a period.
7. is_R_capitalized: A binary feature indicating if the right token is capitalized.
8. is_R_a_number: A binary feature indicating if the right token is a number.

'''

def get_L(L_token):
    return L_token[:-1] if L_token.endswith(".") else 0

def get_R(R_token):
    return R_token if R_token  != '' else 0

def is_L_less_than_four(L_token):
    L = L_token[:-1]
    return 1 if len(L) < 4 else 0

def is_L_a_number(L_token):
    L = L_token[:-1]
    return 1 if L.isdigit() else 0

def is_L_capitalized(L_token):
    L = L_token[:-1]

    if L == '':
        return 0
    
    return 1 if L[0].isupper() else 0

def count_L(L_token):
    L = L_token[:-1]
    return L_counts[L] if L in L_counts else 0

def is_R_capitalized(R_token):
    if R_token != '':
        return 1 if R_token[0].isupper() else 0
    
    return 0

def is_R_a_number(R_token):
    if R_token != '':
        return 1 if R_token.isdigit() else 0
    
    return 0

'''
The extract_features function extracts the features from the tokens to the left and right of the period 
and returns them as a list. Each individual feature is intentionally kept as a separate function to 
allow for easy modification and testing of individual features.

'''
def extract_features(L_token, R_token):
    return [
        get_L(L_token),
        get_R(R_token),
        is_L_less_than_four(L_token),
        is_L_a_number(L_token),
        is_L_capitalized(L_token),
        count_L(L_token),
        is_R_capitalized(R_token),
        is_R_a_number(R_token)
    ]

def encode_tokens(x_train):
    L_encoding = {}
    R_encoding = {}

    next_L_id = 0
    next_R_id = 1   # Start R encoding from 1. 0 is reserved for the empty token.

    for feature_vector in x_train:
        L_token = feature_vector[0]
        R_token = feature_vector[1]

        if L_token not in L_encoding:
            L_encoding[L_token] = next_L_id
            next_L_id += 1

        feature_vector[0] = L_encoding[L_token]

        if R_token == 0:
            continue

        if R_token not in R_encoding:
            R_encoding[R_token] = next_R_id
            next_R_id += 1

        feature_vector[1] = R_encoding[R_token]

    #print("L Encoding:", L_encoding)
    #print("R Encoding:", R_encoding)

def encode_labels(y_train):
    label_encoding = {
        "NEOS": 0,
        "EOS": 1
    }

    for i, label in enumerate(y_train):
        y_train[i] = label_encoding[label]

with open(train_file, "r") as file:
    lines = file.readlines()

    L_counts = {} # Dictionary to store the counts of L tokens

    # First Pass of the data: Count the occurrences of each L token in the training data
    for line in lines:
        columns = line.split()
        token_number = columns[0]
        token = columns[1]
        #label = columns[2]

        if token.endswith("."):
            L = token[:-1]

            if L in L_counts:
                L_counts[L] += 1
            else:
                L_counts[L] = 1

    # Second Pass of the data: Extract features for each token in the training data
    for i, line in enumerate(lines):
        columns = line.split()
        L_token = columns[1]

        if L_token.endswith("."):
            if i + 1 < len(lines):
                R_token = lines[i + 1].split()[1]
            else:
                R_token = ''

            X_train.append(extract_features(L_token, R_token)) # Append the extracted features to the X_train list
            Y_train.append(columns[2]) # Append the label to the Y_train list

encode_tokens(X_train) # Encode the L and R tokens in the X_train list
encode_labels(Y_train) # Encode the labels in the Y_train list

#print(X_train)
#print(Y_train)
            