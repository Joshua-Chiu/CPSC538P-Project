import pickle

# Specify the path to your .pkl file
file_path = 'triplets_test.pkl'

# Open and load the .pkl file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Display the loaded data
print(data)
