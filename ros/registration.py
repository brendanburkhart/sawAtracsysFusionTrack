import math
import numpy as np
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt

# index in reference of nearest neighbor of each point in points
# if unique, then returns None if matching isn't one-to-one
def nearest_neighbor(reference, points, unique=True):
    # each record has n poses but we don't know if they are sorted by markers
    # find correspondence to reference marker that minimizes pair-wise distance
    correspondence = scipy.spatial.distance.cdist(points, reference).argmin(axis=0)

    # skip records where naive-correspondence isn't one-to-one
    if unique and len(np.unique(correspondence)) != len(points):
        return None

    return correspondence

# find Euclidean transformation to align points to reference
# uses Kabsch algorithm, solution minimizes RMS error
# note: reference and points must be in same order
def align(reference, points):
    # Align centroids to remove translation
    reference_centroid = np.mean(reference, axis=0)
    points_centroid = np.mean(points, axis=0)
    reference_centered = reference - reference_centroid
    points_centered = points - points_centroid

    covariance = np.matmul(reference_centered.T, points_centered)
    u, s, vh = np.linalg.svd(covariance)
    d = math.copysign(1, np.linalg.det(np.matmul(u, vh)))
    C = np.diag([1, 1, d])

    R = np.matmul(u, np.matmul(C, vh))
    t = reference_centroid - np.matmul(R, points_centroid)

    return R, t     

# root mean square error, where A is R^(kxnxm) and B is R^(nxm)
def rms_error(A, B):
    error = A - B
    norm = np.linalg.norm(error, axis=len(error.shape)-1)
    return np.sqrt(np.mean(norm**2))

# classical ICP
# determine correspondence order between reference and record
# once correspondence is known, use Kabsch algorithm to check RMS error of alignment
def iterative_closest_point(reference, record, iterations=20, initial_order=False):
    aligned_record = np.copy(record)

    for i in range(iterations):
        if initial_order and i == 0:
            ordered_record = np.copy(aligned_record)
        else:
            order = nearest_neighbor(reference, aligned_record, unique=False)
            ordered_record = aligned_record[order, :]

        R, t = align(reference, ordered_record)
        ordered_record = np.matmul(ordered_record, R.T) + t
        aligned_record = np.matmul(aligned_record, R.T) + t

    ordered_record = record[order]
    R, t = align(reference, ordered_record)
    aligned_record = np.matmul(ordered_record, R.T) + t

    error = rms_error(reference, aligned_record)
    return order, (R, t), error

# find nearest distinct pair of points
# return indices [i, j], and distance
def nearest_pair(points):
    # nearest pair corresponds to smallest off-diagonal distance
    distances = scipy.spatial.distance.cdist(points, points)
    diagonal_mask = np.eye(*distances.shape, dtype=bool)
    distances = np.where(diagonal_mask, np.inf, distances)
    nearest_pair = np.unravel_index(distances.argmin(), distances.shape)
    distance = distances.min()

    return np.array(nearest_pair), distance

def conformal_metric_m(n):
    M = np.zeros((n+2, n+2))
    M[0:n, 0:n] = np.eye(n)
    M[n, n+1] = -1
    M[n+1, n] = -1

    return M

# compute metric for conformal R^n+1,1 space
def conformal_metric(p, q):
    n = len(p) - 2
    assert n == len(q) - 2

    # p.q = pTMq
    return p.T @ conformal_metric_m(n) @ q

def conformal_projection(p):
    l = np.linalg.norm(p)
    x = np.append(p, [1, 0.5*(l*l)])
    return x

def extract_conformal_sphere(x):
    alpha = 1/x[-2]
    y = alpha*x
    center = y[0:-2]
    r2 = np.linalg.norm(center)**2 - 2*y[-1]
    radius = np.sqrt(r2)

    return center, radius

# See "Total Least Squares Fitting of k-Spheres[...]" by L Dorst
# Fits k-sphere to (k+1) dimensional points
def k_sphere_fit(points):
    N, n = points.shape
    D = np.array([conformal_projection(p) for p in points])
    P = (D.T @ D @ conformal_metric_m(n)) / N

    eigenvalues, eigenvectors = scipy.linalg.eig(P)
    epsilon = np.finfo(np.float32).eps
    index = np.where(eigenvalues >= -epsilon, eigenvalues, np.inf).argmin()

    x = eigenvectors[:, index]
    center, radius = extract_conformal_sphere(x.real)
    
    error = np.sqrt(np.mean([(np.linalg.norm(p - center) - radius)**2 for p in points]))

    return center, radius, error

def determine_pivot(marker_samples):
    _, marker_count, _ = marker_samples.shape
    spheres = [k_sphere_fit(marker_samples[:, i, :]) for i in range(marker_count)]
    total_distance = sum([radius for _, radius, _ in spheres])
    weights = np.array([radius/total_distance for _, radius, _ in spheres])
    centers = np.array([center for center, _, _ in spheres])
    error = np.sqrt(np.mean([error**2 for _, _, error in spheres]))
    pivot = np.average(centers, axis=0, weights=weights)

    return pivot, error

def simulate(center, radius, noise=0.2, count=60):
    measurements = []

    for _ in range(count):
        direction = np.random.normal([0, 0, 0], [12, 12, 12])
        direction = direction/np.linalg.norm(direction)
        error = np.random.normal(np.zeros(3), noise*np.ones(3))
        point = center + radius*direction + error
        measurements.append(point + error)

    measurements = np.array(measurements)
    m_center, m_radius, m_error = k_sphere_fit(measurements)
    return np.linalg.norm(m_center - center), np.linalg.norm(m_radius - radius), m_error
