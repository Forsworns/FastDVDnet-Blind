import os

for root, _, files in os.walk('test_A'):
    for file in files:
        old_name = os.path.join(root, file)
        new_file = file.rstrip('.png').zfill(4)+'.png'
        new_name = os.path.join(root, new_file)
        os.rename(old_name, new_name)

