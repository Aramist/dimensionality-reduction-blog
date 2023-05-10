from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import linprog
from scipy.spatial import KDTree
from sklearn.manifold import MDS, TSNE
from umap import UMAP


##########################################
# Stuff for plotting
##########################################

def helper_plot_err(points, embed_points, embed_type, fig_title, fname, num_nn=15):
    cos_sims = iou_distance(points, embed_points, num_nn)
    
    plt.figure(figsize=(12, 5))
    plt.suptitle(fig_title, fontsize=16)
    plt.subplot(121)
    plt.title(f'{embed_type} embedding')
    plt.scatter(embed_points[:, 0], embed_points[:, 1], c=cos_sims, cmap='magma_r', s=3, alpha=1)
    plt.clim(0, 1)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(122)
    plt.title('Neighborhood IOU Scores')
    plt.hist(cos_sims)
    plt.xlim(0, 1)
    if fname is not None:
        plt.savefig(fname, dpi=240)
    plt.show()

##########################################
# Stuff for cloth
##########################################

def load_obj_vertices(fname: str) -> np.ndarray:
    """ Loads vertices from an .obj file. Assumes only one object is present

    Args:
        fname (str): File name

    Returns:
        np.ndarray: Vertices extracted from the file
    """
    with open(fname, 'r') as ctx:
        lines = ctx.readlines()
    vertices = []
    for l in lines:
        if not l.startswith('v '):
            continue
        coords = list(map(lambda x: float(x), l.split(' ')[1:]))
        coords = np.array(coords)
        vertices.append(coords)
    vertices = np.stack(vertices, axis=0)
    return vertices

def map_cloth_points(points: np.ndarray, from_map: np.ndarray, to_map: np.ndarray) -> np.ndarray:
    """ Helper function to map points from the 2d unit square to a model of a cloth
    dropped over some rigid objects.

    Returns:
        ndarray: An (n, 3) array of the provided points mapped to the deformed cloth
    """
    new_max, new_min = 0.995, -0.995
    points = points * (new_max - new_min) + new_min  # shift range from [0, 1] to [-0.995, 0.995]
    flat_map = from_map[:, (0, 2)]  # drop height dimension, plane is still flat
    point_mag = (points ** 2).sum(axis=1)
    map_mag = (flat_map ** 2).sum(axis=1)
    dist_mat = (
        point_mag[:, None] + map_mag[None, :]
        - 2 * np.einsum('ni,mi->nm', points, flat_map)
    )
    nearest_triangle_idx = np.argsort(dist_mat, axis=1)[:, :3]
    
    # get barycentric coords for each source point
    flat_triangles = flat_map[nearest_triangle_idx]
    dest_triangles = to_map[nearest_triangle_idx]
    
    # point[i][0] ~ a * flat_triangle[i][0] + b * flat_triangle[i][1] + c * flat_triangle[i][2]
    # ect
    # point[i] (2x1 mat) ~ triangle_pts (2x3 stacked) @ [a; b; c] (3x1 stacks params
    bary_points = []
    good_idx = []
    for n, ((u, v), flat_triangle) in enumerate(zip(points, flat_triangles)):
        bary_coords = linprog(
            c=np.ones((3,)),
            A_eq=np.concatenate([flat_triangle.T, np.ones((1, 3))], axis=0),
            b_eq=[u, v, 1],
            # bounds=(0, 1),
        )
        if bary_coords.x is None:
            # boundary points tend to fail
            continue
        bary_points.append(bary_coords.x)
        good_idx.append(n)
    bary_points = np.stack(bary_points, axis=0)
    
    # dest triangles: (n, 3 (num points), 3 (xyz))
    # bary_points: (n, 3 (num vertices))
    dest_points = np.einsum('nvc,nv->nc', dest_triangles[good_idx], bary_points)
    return dest_points

##########################################
# Stuff for distances
##########################################

def dense_pairwise_distances(points):
    # points: (n, d)
    # memory saving identity: (a-b) ** 2 = a**2 - 2ab + b**2
    mags = (points ** 2).sum(axis=1)
    dists = (
        mags[None, :]
        + mags [:, None]
        - 2 * np.einsum('ad,bd->ab', points, points)
    )
    return dists

def cross_dist(points, targets):
    # points: (n, dim) array of points
    # targets: (n, k, dim) array of targets
    # distances will be computed between points_i and every point in targets_i
    n, k, dim = targets.shape
    targets = targets.reshape(n*k, dim)
    
    exp_points = np.repeat(points, k, axis=0)
    mags_points = np.repeat((points * points).sum(axis=1), (k,))
    mags_targets = (targets * targets).sum(axis=1)

    dists = (
        mags_points
        + mags_targets
        - 2 * np.einsum('nd,nd->n', exp_points, targets)
    )
    dists = dists.reshape(n, k)
    return dists

def get_top_nn(points, k):
    tree = KDTree(points)
    nn_idx = np.empty((len(points), k))
    for p in range(len(points)):
        _, nearest = tree.query(points[p], k + 1)
        nearest = nearest[1:]  # the nearest point should be the queried point itself
        nn_idx[p] = nearest
    return nn_idx.astype(int)


def embed_in_high_dim(points, embed_dim):
    manifold_dim = points.shape[1]
    embed_ortho = np.random.randn(manifold_dim, embed_dim)
    # make the columns orthogonal
    embed_ortho[0] /= np.linalg.norm(embed_ortho[0])
    for dim in range(1, manifold_dim):
        # project [1] onto [0]
        for ref in range(0, dim):
            embed_ortho[dim] -= np.dot(embed_ortho[ref], embed_ortho[dim]) * embed_ortho[ref]
            # normalizing at every step helps avoid div by zero
        embed_ortho[dim] /= np.linalg.norm(embed_ortho[dim])

    points = np.einsum('me,nm->ne', embed_ortho, points)
    return points

##########################################
# Embedding methods
##########################################

def make_tsne_embedding(points, **tsne_settings):
    tsne_params = {  # more or less default params
        'perplexity': 50,
        'early_exaggeration': 12,
        # 'method': 'exact',
    }
    tsne_params.update(tsne_settings)
    return TSNE(**tsne_params).fit_transform(points)

def make_pca_embedding(points):
    return PCA().fit_transform(points)[:, :2]

def make_mds_embedding(points):
    return MDS().fit_transform(points)

def make_umap_embedding(points, **umap_settings):
    return UMAP(**umap_settings).fit_transform(points)

##########################################
# Distance methods
##########################################

def nn_cos_sim_distance(points_a, points_b, num_nn):
    top_nn_idx = get_top_nn(points_a, num_nn)
    orig_dist = cross_dist(points_a, points_a[top_nn_idx])
    mod_dist = cross_dist(points_b, points_b[top_nn_idx])
    amag = 1 / np.linalg.norm(orig_dist, axis=1)
    bmag = 1 / np.linalg.norm(mod_dist, axis=1)
    cos_sims = np.einsum('ij,ij,i,i->i', orig_dist, mod_dist, amag, bmag)
    return cos_sims

def iou_distance(points_a, points_b, num_nn):
    top_nn_a = [set(nums) for nums in get_top_nn(points_a, num_nn)]
    top_nn_b = [set(nums) for nums in get_top_nn(points_b, num_nn)]
    intersections = np.array([len(a.intersection(b)) for a,b in zip(top_nn_a, top_nn_b)])
    unions = np.array([len(a.union(b)) for a,b in zip(top_nn_a, top_nn_b)])
    return intersections / unions


##########################################
# Sampling methods
##########################################

def make_gaussian(mean: Union[list, np.ndarray], cov: Union[float, np.ndarray], resolution: int=300) -> np.ndarray:
    """Renders a 2d gaussian on the unit square [0,1), [0, 1) with the given parameters

    Args:
        mean (Union[list, np.ndarray])
        cov (Union[float, np.ndarray]): Covariance. Isotropic if a scalar is provided. 
            Axis-aligned if two values are provided
        resolution (int, optional): Size of the grid generated. Defaults to 300.

    Returns:
        np.ndarray: The unscaled gaussian evaluated at evenly spaced points along the unit square
    """
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(cov, np.ndarray):
        cov = np.array(cov)
    if len(cov.shape) == 0:
        cov = np.array([[cov**2, 0], [0, cov**2]])
    elif len(cov.shape) == 1:
        cov = np.array([[cov[0]**2, 0], [0, cov[1]**2]])
    # mean: (2,)
    # cov: (2, 2)
    # resolution: scalar
    
    bins = np.linspace(0, 1, resolution)
    points = np.stack(np.meshgrid(bins, bins, indexing='xy'), axis=2).reshape(-1, 2)
    
    prec = np.linalg.inv(cov)
    diffs = (points - mean[None, :])
    #-0.5 (x-mu)^t prec (x-mu)
    exp_term = -0.5 * np.einsum('ni,ij,nj -> n', diffs, prec, diffs)
    exp_term -= exp_term.max()
    return np.exp(exp_term).reshape(resolution, resolution)

def sample_plane(num_points: int, pmf: Optional[np.ndarray]=None) -> np.ndarray:
    """Samples points from the unit square with the provided density

    Args:
        num_points (int): number of points to sample
        pmf (Optional[np.ndarray], optional): Density to sample from. Defaults to uniform density.

    Returns:
        np.ndarray: Points sampled from unit square
    """
    if pmf is None:
        return np.random.random(size=(num_points, 2))
    pmf = pmf / pmf.sum()
    flat_pmf = pmf.ravel()
    n_yb, n_xb = pmf.shape
    x_bins = np.linspace(0, 1, n_xb + 1, endpoint=True)
    y_bins = np.linspace(0, 1, n_yb + 1, endpoint=True)[::-1]

    # first sample a bin
    bins = np.random.choice(np.arange(len(flat_pmf)), size=num_points, replace=True, p=flat_pmf)
    # within each bin, randomly sample
    points = []
    for bin in bins:
        bin = np.unravel_index(bin, pmf.shape)  # returns a 2-tuple
        x_lo, x_hi = x_bins[bin[1]:bin[1]+2]
        y_lo, y_hi = y_bins[bin[0]:bin[0]+2]
        x = np.random.uniform(x_lo, x_hi)
        y = np.random.uniform(y_lo, y_hi)
        points.append(np.array([x, y]))
    return np.stack(points)


def sample_sphere(manifold_dim, num_points, snr):    
    rand_vecs = np.random.randn(num_points, manifold_dim)
    rand_vecs /= np.linalg.norm(rand_vecs, axis=1, keepdims=True)  # project to sphere
    noise = np.random.randn(num_points, manifold_dim)
    lowd_points = rand_vecs + noise / snr
    return lowd_points


def sample_hypercube(manifold_dim, num_points, snr):
    # Samples from surface of hypercube
    rand_vecs = np.random.uniform(-1, 1, size=(num_points, manifold_dim))
    rand_signs = np.sign(rand_vecs)
    
    if manifold_dim < 2:
        raise ValueError('Manifold dim should be at least 2')
    
    # maybe there is a better way to do this?
    # in binary, this mask corresponds to randomly selecting any possibility except all Falses and all Trues
    # When a bit is 1, that dim should be replaced by its sign
    mask = np.random.randint(1, 2**manifold_dim - 1, size=(num_points,))
    helper = 2**np.arange(manifold_dim)
    mask = np.nonzero(mask[:, None] & helper[None, :])  # Now a true/false mask for dims
    
    rand_vecs[mask] = rand_signs[mask]
    
    noise = np.random.randn(num_points, manifold_dim)
    lowd_points = rand_vecs + noise / snr
    return lowd_points

def sample_torus(num_points, snr, tube_radius, outer_radius):
    # adapted from https://rdrr.io/cran/TDA/src/R/torusUnif.R
    # theta is angle within tube
    # phi is angle around donut
    # to avoid having higher density along inside of donut, theta is sampled non-uniformly
    thetas = []
    while len(thetas) < num_points:
        x = np.random.uniform(0, 2 * np.pi)
        y = np.random.uniform(0, 1 / np.pi)
        fx = (1 + (tube_radius / outer_radius) * np.cos(x)) / (2 * np.pi)
        if y < fx:
            thetas.append(x)
    thetas = np.array(thetas)
    phis = np.random.uniform(0, 2 * np.pi, (num_points,))
    coord_x = (outer_radius + tube_radius * np.cos(thetas)) * np.cos(phis)
    coord_y = (outer_radius + tube_radius * np.cos(thetas)) * np.sin(phis)
    coord_z = tube_radius * np.sin(thetas)
    points = np.stack([coord_x, coord_y, coord_z], axis=1)
    
    noise = np.random.randn(num_points, 3)
    points = points + noise / snr
    return points

def sample_cloth(num_points: int, pmf: Optional[np.ndarray] = None) -> np.ndarray:
    """Samples from the deformed cloth manifold with the provided density

    Args:
        num_points (int): number of points to sample
        pmf (Optional[np.ndarray], optional): The desired density. Defaults to a uniform density.

    Returns:
        np.ndarray: 3-dimensional points sampled from the deformed cloth manifold
    """
    uv_points = sample_plane(num_points, pmf)
    init_vertices = load_obj_vertices('cloth_images/init_cloth.obj')
    deformed_vertices = load_obj_vertices('cloth_images/deformed_cloth.obj')
    return map_cloth_points(uv_points, init_vertices, deformed_vertices)