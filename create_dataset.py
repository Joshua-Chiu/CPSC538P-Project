import os

def generate_coords(path):
    print(path)
    pass

PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PATH, "entireid")

for item in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, item)
    
    # Check if the item is a directory (subfolder)
    if os.path.isdir(folder_path):
        print(f"Subfolder: {item}")
        
        # List files in the subfolder
        files = os.listdir(folder_path)
        for file in files:
            if file.lower().endswith(".jpg"):
                generate_coords(os.path.join(folder_path, file))
