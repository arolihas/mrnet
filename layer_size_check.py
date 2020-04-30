import numpy as np
import os

# Specify the .dcm folder path
folder_path = "MRNet-v1.0\\valid\\coronal"
files = os.listdir(folder_path)

max = 0
for n, filename in enumerate(files):
    if ".npy" in filename:
        arr = np.load((os.path.join(folder_path, filename)))
        val = (arr.shape[0])
        if max < val:
            print(val)
            max = val

print("Max is : " + str(max))

# Train

# Max Layers - axial : 61
# Max Layers - coronal : 58
# Max Layers - sagittal : 51

# Valid

# Max Layers - axial : 52
# Max Layers - coronal : 48
# Max Layers - sagittal : 45
