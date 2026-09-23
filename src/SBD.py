import sys
from sklearn.tree import DecisionTreeClassifier, export_text

'''
Program Pipline:
1. Periods in the training dataset are identified
2. Feature vectors are built for the training dataset and inserted into an array X
3. Corresponding labels are inserted into an array Y
4. Feature vectors in array X and labels in array Y are encoded
5. Period in the test dataset are indentified
6. Feature vectors are built for the test dataset and inserted into an array X
7. Array X and Array Y are passed to the classifer for training
8. The fitted model predicts labels
9. Decision-tree graph is generated
10. Model accuracy is calculated
11. Text file comparing gold-standard labels with predicted labels is generated

'''

train_file = "../data/train/" + sys.argv[1]
test_file = "../data/test/" + sys.argv[2]

if len(sys.argv) != 3 and not sys.argv[1].endswith(".train") and not sys.argv[2].endswith(".test"):
    print("Usage: python SBD.py <train_file> <test_file>")
    sys.exit(1)

# Dictionaries storing token encodings
L_encoding = {}
R_encoding = {}

# Unkown values for tokens that appear in test dataset but not in training dataset
UNK_L = 999999
UNK_R = 999999

# Feature names and class names used to enhance graph readability
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

'''
Functions to extract features from the tokens. The features are:
(Core Features)
1. L: The token to the left of the period (without the period).
2. R: The token to the right of the period.
3. is_L_less_than_four: A binary feature indicating if the left token is less than four characters long.
4. is_L_a_number: A binary feature indicating if the left token is a number.
5. is_L_capitalized: A binary feature indicating if the left token is capitalized.

(Additional Features)
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

'''
The encode_tokens and encode_labels function encode the feature vectors in both the vector array and label array.
Dictionaries are used to store and keep track of the token-value pairs. The encode_tokens function utilizes a binary flag
to first build the dictionaries using the training dataset. The test dataset feature vectors are then encoded by setting the flag 
to 'False' using the already built dictionary to keep token encoding consistent across the training and test datasets.

'''

def encode_tokens(X_array, build_encoding=True):
    if build_encoding:

        for feature_vector in X_array:
            L_token = feature_vector[0]
            R_token = feature_vector[1]

            if L_token not in L_encoding:
                L_encoding[L_token] = len(L_encoding)  # Assign the next available ID based on the current size of the encoding dictionary

            feature_vector[0] = L_encoding[L_token]

            if R_token == 0:
                continue

            if R_token not in R_encoding:
                R_encoding[R_token] = len(R_encoding) + 1 # Increment by 1 to reserve 0 for the empty token

            feature_vector[1] = R_encoding[R_token]

    else:
        for feature_vector in X_array:
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

def encode_labels(Y_array):
    label_encoding = {
        "NEOS": 0,
        "EOS": 1
    }

    for i, label in enumerate(Y_array):
        Y_array[i] = label_encoding[label]

'''
The function build_L_counts performs a pass-through of the training data file. This function counts and stores all tokens
that appear to the left of a period. This number can give the model statistical insight into the entire training dataset.
This also serves an alternative to using pre-loaded dictionary of common abbreviations.

'''

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

'''
The function build_feature_array builds the feature arrays. This function removes this responsibilty
from the preprocess_data function.

'''

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

                X_array.append(extract_features(L_token, R_token, L_counts)) 
                Y_array.append(columns[2])

'''
The preprocess_data function handles both the training and test datasets. Feature vectors are built and inserted into arrays X and Y so they
can be properly passed to the decision tree classifer. The arrays are also encoded here.

'''

def preprocess_data(train_file, test_file):
    X_train = [] 
    Y_train = []

    X_test = []
    Y_test = []

    L_counts = build_L_counts(train_file)

    build_feature_array(train_file, X_train, Y_train, L_counts)  # Build the feature array for the training data
    
    encode_tokens(X_train)
    encode_labels(Y_train)

    build_feature_array(test_file, X_test, Y_test, L_counts)  # Build the feature array for the test data using the same L_counts from the training data

    encode_tokens(X_test, build_encoding=False) # Encode the L and R tokens in the X_test list using the existing encoding
    encode_labels(Y_test)   

    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test =  preprocess_data(train_file, test_file) # Preprocess the training and test data and extract features


# Debugging funtions (ignore)
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
                   