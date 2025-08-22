import torch
import numpy as np

outputs_triton = np.load("outputs_triton.npy")
encoded_python = np.load("encoded_python.npy")

for i in range(751): 
    print(f"{outputs_triton[0][0][i]}\t{encoded_python[0][0][i]}")

# breakpoint()

min_len = min(outputs_triton.shape[-1], encoded_python.shape[-1])

vec1 = outputs_triton[:, :, :min_len]
vec2 = encoded_python[:, :, :min_len]

distance = np.linalg.norm(vec1 - vec2)
print("Euclidean distance:", distance)
