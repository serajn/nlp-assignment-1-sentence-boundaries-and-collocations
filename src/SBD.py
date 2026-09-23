import sys
from sklearn.tree import DecisionTreeClassifier, export_text

train_file = "../data/train/" + sys.argv[1]
test_file = "../data/test/" + sys.argv[2]

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

def count_L(L_token, L_counts):
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
def extract_features(L_token, R_token, L_counts):
    return [
        get_L(L_token),
        get_R(R_token),
        is_L_less_than_four(get_L(L_token)),
        is_L_a_number(get_L(L_token)),
        is_L_capitalized(get_L(L_token)),
        count_L(L_token, L_counts),
        is_R_capitalized(R_token),
        is_R_a_number(R_token)
    ]

L_encoding = {}
R_encoding = {}

UNK_L = 999999
UNK_R = 999999

def encode_tokens(x_array, build_encoding=True):

    #next_L_id = 0
    #next_R_id = 1   # Start R encoding from 1. 0 is reserved for the empty token.

    if build_encoding:

        for feature_vector in x_array:
            L_token = feature_vector[0]
            R_token = feature_vector[1]

            if L_token not in L_encoding:
                #L_encoding[L_token] = next_L_id
                #next_L_id += 1

                L_encoding[L_token] = len(L_encoding)  # Assign the next available ID based on the current size of the encoding dictionary

            feature_vector[0] = L_encoding[L_token]

            if R_token == 0:
                continue

            if R_token not in R_encoding:
                #R_encoding[R_token] = next_R_id
                #next_R_id += 1

                R_encoding[R_token] = len(R_encoding) + 1

            feature_vector[1] = R_encoding[R_token]

    else:
        for feature_vector in x_array:
            L_token = feature_vector[0]
            R_token = feature_vector[1]

            if L_token in L_encoding:
                feature_vector[0] = L_encoding[L_token]
            else:
                feature_vector[0] = UNK_L

            if R_token == 0:
                continue

            if R_token in R_encoding:
                feature_vector[1] = R_encoding[R_token]
            else:
                feature_vector[1] = UNK_R

    #print("L Encoding:", L_encoding)
    #print("R Encoding:", R_encoding)

def encode_labels(y_train):
    label_encoding = {
        "NEOS": 0,
        "EOS": 1
    }

    for i, label in enumerate(y_train):
        y_train[i] = label_encoding[label]

def build_L_counts(data_file):
    with open(data_file, "r") as file:
        L_counts = {}

        lines = file.readlines()

        for line in lines:
            columns = line.split()
            token = columns[1]

            if token.endswith("."):
                L = token[:-1]

                if L in L_counts:
                    L_counts[L] += 1
                else:
                    L_counts[L] = 1

    return L_counts

def build_feature_array(data_file, X_array, Y_array, L_counts):
    with open(data_file, "r") as file:
        lines = file.readlines()

        for i, line in enumerate(lines):
            columns = line.split()
            L_token = columns[1]

            if L_token.endswith("."):
                if i + 1 < len(lines):
                    R_token = lines[i + 1].split()[1]
                else:
                    R_token = ''

                X_array.append(extract_features(L_token, R_token, L_counts)) # Append the extracted features to the X_train list
                Y_array.append(columns[2]) # Append the label to the Y_train list


def preprocess_data(train_file, test_file):
    X_train = [] 
    Y_train = []

    X_test = []
    Y_test = []

    L_counts = build_L_counts(train_file)  # Build the L_counts dictionary for the training data

    build_feature_array(train_file, X_train, Y_train, L_counts)  # Build the feature array for the training data
    
    encode_tokens(X_train) # Encode the L and R tokens in the X_train list
    encode_labels(Y_train) # Encode the labels in the Y_train list

    build_feature_array(test_file, X_test, Y_test, L_counts)  # Build the feature array for the test data using the same L_counts from the training data

    encode_tokens(X_test, build_encoding=False) # Encode the L and R tokens in the X_test list using the existing encoding
    encode_labels(Y_test) # Encode the labels in the Y_test list   

    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test =  preprocess_data(train_file, test_file) # Preprocess the training and test data and extract features

'''
with open("encoded_train.txt", "w") as file:
    for feature_vector in X_train:
        file.write(str(feature_vector) + "\n")

with open("L_encoding.txt", "w") as file:
    for token, encoding in L_encoding.items():
        file.write(f"{token}: {encoding}\n")

with open("encoded_test.txt", "w") as file:
    for feature_vector in X_test:
        file.write(str(feature_vector) + "\n")

with open("R_encoding.txt", "w") as file:
    for token, encoding in R_encoding.items():
        file.write(f"{token}: {encoding}\n")

'''

classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test) # Make predictions on the test data

feature_names = [
    "L",
    "R",
    "is_L_less_than_four",
    "is_L_a_number",
    "is_L_capitalized",
    "count_L",
    "is_R_capitalized",
    "is_R_a_number"
]

class_names = ["NEOS", "EOS"]

tree_text = export_text(classifier, feature_names=feature_names, class_names=class_names)
print(tree_text) # Print the decision tree structure

correct = 0

for prediction, actual in zip(predictions, Y_test):
    if prediction == actual:
        correct += 1

accuracy = correct / len(Y_test) * 100

print(f"Accuracy: {accuracy:.2f}%") # Print the accuracy of the model on the test data


i = 0

with open(test_file, "r") as infile, open("SBD.test.out", "w") as outfile:
    for line in infile:
        columns = line.split()
        token = columns[1]
        label = columns[2]

        if token.endswith("."):
            predicted_label = predictions[i]
            if predicted_label == 0:
                predicted_label = "NEOS"
            else:
                predicted_label = "EOS"
            i += 1

            outfile.write(f"{token} {label} {predicted_label}\n")
                   