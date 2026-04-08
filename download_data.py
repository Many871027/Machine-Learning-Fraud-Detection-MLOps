import kagglehub
import shutil
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Downloaded to:", path)

source_file = os.path.join(path, "creditcard.csv")
dest_file = "creditcard.csv"

if os.path.exists(source_file):
    shutil.copy(source_file, dest_file)
    print(f"Successfully copied 'creditcard.csv' to current directory.")
else:
    print(f"File not found in {path}. Contents:")
    for root, dirs, files in os.walk(path):
        for name in files:
            print(os.path.join(root, name))
