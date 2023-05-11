import time

import numpy as np

from blog_utils import *
'''
nonuniform_pmf = (0.07
    + make_gaussian([0.65, 0.75], 0.05, 200)
    + make_gaussian([0.25, 0.55], [0.10, 0.07], 200)
    + make_gaussian([0.85, 0.15], [0.05, 0.2], 200)
)
nonuniform_pmf /= nonuniform_pmf.sum()

nonuniform_sample = sample_cloth(2000, nonuniform_pmf)
uniform_sample = sample_cloth(2000)

# No noise for this simulation
nonuniform_sample = embed_in_high_dim(nonuniform_sample, 10)
uniform_sample = embed_in_high_dim(uniform_sample, 10)
'''
# Save these because we need to compare the embeddings to them to get the error
# np.save('cloth_images/nonuniform_sample.npy', nonuniform_sample)
# np.save('cloth_images/uniform_sample.npy', uniform_sample)
print("Created and saved samples from cloth manifold")

uniform_sample = np.load('cloth_images/uniform_sample.npy')
nonuniform_sample = np.load('cloth_images/nonuniform_sample.npy')

# Generate all the embeddings...
'''
start_time = time.time()

umaps_uniform = {}
for n in (2, 5, 10, 20, 50, 100, 200):
    for min_dist in (0.0, 0.1, 0.25, 0.5, 0.65, 0.8, 0.99):
        umaps_uniform[f'{n}_{min_dist:.2f}'] = make_umap_embedding(uniform_sample, n_neighbors=n, min_dist=min_dist)

umaps_nonuniform = {}
for n in (2, 5, 10, 20, 50, 100, 200):
    for min_dist in (0.0, 0.1, 0.25, 0.5, 0.65, 0.8, 0.99):
        umaps_nonuniform[f'{n}_{min_dist:.2f}'] = make_umap_embedding(nonuniform_sample, n_neighbors=n, min_dist=min_dist)

for k, v in umaps_uniform.items():
    np.save(f'uniform_cloth_umaps/{k}.npy', v)
for k, v in umaps_nonuniform.items():
    np.save(f'nonuniform_cloth_umaps/{k}.npy', v)

print(f"Generated and saved all UMAP embeddings in {time.time() - start_time:.2f} seconds")
'''
start_time = time.time()
tsnes_uniform = {}
for p in (5, 15, 25, 45, 65, 85, 105):
    for ee in np.linspace(6, 42, 7, endpoint=True):
        tsnes_uniform[f'{p}_{ee:.1f}'] = make_tsne_embedding(uniform_sample, perplexity=p, early_exaggeration=ee)

tsnes_nonuniform = {}
for p in (5, 15, 25, 45, 65, 85, 105):
    for ee in np.linspace(6, 42, 7, endpoint=True):
        tsnes_nonuniform[f'{p}_{ee:.1f}'] = make_tsne_embedding(nonuniform_sample, perplexity=p, early_exaggeration=ee)

for k, v in tsnes_uniform.items():
    np.save(f'uniform_cloth_tsnes/{k}.npy', v)
for k, v in tsnes_nonuniform.items():
    np.save(f'nonuniform_cloth_tsnes/{k}.npy', v)

print(f"Generated and saved all t-SNE embeddings in {time.time() - start_time:.2f} seconds")