#1
import numpy as np
import plotly.express as px
import time

#2
vectors = 2 * np.random.random_sample((1000, 3)) - 1

#3
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#4
cosine_similarities = np.zeros((len(vectors), len(vectors)))

start_time = time.time()

for i, vec1 in enumerate(vectors):
    for j, vec2 in enumerate(vectors):
        cosine_similarities[i, j] = cosine_similarity(vec1, vec2)

print(f"Calculation took: {time.time() - start_time} seconds.")

fig = px.imshow(cosine_similarities)
fig.update_xaxes(side = "top")
fig.show()

#5
print(f'vectors is already a matrix of 1000x3 dims: {vectors.shape}')

#6
start_time = time.time()

l2_norms = np.apply_along_axis(np.linalg.norm, 1, vectors)

#7
norm_matrix = vectors / l2_norms[:, None]

#8
cosine_similarities_2 = np.matmul(norm_matrix, norm_matrix.T)

print(f"Calculation took: {time.time() - start_time} seconds.")

fig = px.imshow(cosine_similarities_2)
fig.update_xaxes(side = "top")
fig.show()

#9
#Matrix calculation is much more time efficient than iterating on each 2 vectors in a loop.