from attr import dataclass
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import heapq
from itertools import product
from scipy.spatial import cKDTree

print("Loading STL...")
mesh = trimesh.load_mesh("5-axis-3d-printer\\stl\\branches.stl")

print("voxelising...")
voxel_size = 0.5
voxel_mesh = mesh.voxelized(pitch=voxel_size)

print("filling...")
# fill the interior of the mesh to get a solid voxel representation
voxel_mesh.fill()


voxel_matrix = voxel_mesh.matrix.astype(np.uint8)
print(f"total number of voxels: {np.sum(voxel_matrix)}")
print(f"voxel matrix shape: {voxel_matrix.shape}")


def voxel_distances(voxel_matrix, pitch=1.0, seed_mask=None):
    """
    Shortest-path distances through occupied voxels.

    Parameters
    ----------
    voxel_matrix : 3D boolean array
        True for material, False for empty space.

    pitch : float or length-3 sequence
        Voxel spacing in physical units.
        For example, pitch=0.5 gives distances in mm if pitch is in mm.

    seed_mask : 3D boolean array, optional
        Voxels from which propagation starts.
        Defaults to all occupied voxels in the lowest occupied z-plane.

    Returns
    -------
    distances : 3D float array
        Zero at seeds.
        Finite distances at reachable material voxels.
        np.inf outside the material and in unreachable components.
    """
    occupied = np.asarray(voxel_matrix, dtype=bool)

    if occupied.ndim != 3:
        raise ValueError("voxel_matrix must be three-dimensional.")

    spacing = np.broadcast_to(
        np.asarray(pitch, dtype=float), (3,)
    )

    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("pitch must contain finite positive values.")

    distances = np.full(occupied.shape, np.inf, dtype=float)

    if seed_mask is None:
        seeds = np.zeros_like(occupied)

        occupied_planes = np.flatnonzero(
            occupied.any(axis=(0, 1))
        )

        if occupied_planes.size == 0:
            return distances

        lowest_z = occupied_planes[0]
        seeds[:, :, lowest_z] = occupied[:, :, lowest_z]

    else:
        seeds = np.asarray(seed_mask, dtype=bool)

        if seeds.shape != occupied.shape:
            raise ValueError("seed_mask must match voxel_matrix.shape.")

        if np.any(seeds & ~occupied):
            raise ValueError("Seed voxels must be inside the material.")

        if not np.any(seeds):
            raise ValueError("seed_mask contains no seeds.")

    # Precompute neighbour offsets, costs and corner-cutting checks.
    moves = []

    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue

        cost = float(np.linalg.norm(np.asarray(offset) * spacing))

        # For a diagonal move, require all intermediate voxels in the
        # small axis-aligned block between the endpoints to be occupied.
        #
        # Example: (1, 1, 0) also requires (1, 0, 0) and (0, 1, 0).
        # This is deliberately conservative around corners.
        choices = [
            (0, step) if step != 0 else (0,)
            for step in offset
        ]

        intermediate_offsets = [
            intermediate
            for intermediate in product(*choices)
            if intermediate != (0, 0, 0)
            and intermediate != offset
        ]

        moves.append((offset, cost, intermediate_offsets))

    heap = []

    for i, j, k in np.argwhere(seeds):
        i, j, k = int(i), int(j), int(k)
        distances[i, j, k] = 0.0
        heap.append((0.0, i, j, k))

    heapq.heapify(heap)

    nx, ny, nz = occupied.shape

    while heap:
        distance, i, j, k = heapq.heappop(heap)

        # A better route may have been found since this entry was queued.
        if distance > distances[i, j, k]:
            continue

        for (di, dj, dk), cost, intermediates in moves:
            ni, nj, nk = i + di, j + dj, k + dk

            if not (
                0 <= ni < nx
                and 0 <= nj < ny
                and 0 <= nk < nz
            ):
                continue

            if not occupied[ni, nj, nk]:
                continue

            candidate_distance = distance + cost

            if candidate_distance >= distances[ni, nj, nk]:
                continue

            # Intermediate indices are in bounds whenever both endpoints
            # are in bounds.
            if any(
                not occupied[i + oi, j + oj, k + ok]
                for oi, oj, ok in intermediates
            ):
                continue

            distances[ni, nj, nk] = candidate_distance
            heapq.heappush(
                heap, (candidate_distance, ni, nj, nk)
            )

    return distances

print("calculating distances...")
distance_matrix = voxel_distances(voxel_matrix, pitch=voxel_size)

# set all inf values to -1
distance_matrix[distance_matrix == np.inf] = -1

def split_regions(intersection_points, threshold):
    """Return a list of regions, each containing a list of 3D points."""
    points = np.asarray(intersection_points, dtype=float)

    if points.size == 0:
        return []

    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    regions = []

    for start in range(len(points)):
        if visited[start]:
            continue

        visited[start] = True
        pending = [start]
        indices = []

        while pending:
            current = pending.pop()
            indices.append(current)

            for neighbour in tree.query_ball_point(points[current], threshold):
                if not visited[neighbour]:
                    visited[neighbour] = True
                    pending.append(neighbour)

        regions.append(points[indices].tolist())

    return regions

dist_step = 1
slice_contours = []
for dist in np.arange(0, distance_matrix.max(), dist_step):

    print(f"finding slice contour for distance {dist},  {dist / distance_matrix.max() * 100:.2f}%...")
    # go through every edge between occupied voxels and find edges where one voxel has distance < dist and the other voxel has distance > dist
    # avoid duplicated edges
    edges_containing_dist = set()
    nx, ny, nz = distance_matrix.shape

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                d1 = distance_matrix[i, j, k]
                # Ignore empty or unreachable voxels.
                if not voxel_matrix[i, j, k] or not np.isfinite(d1) or d1 < 0:
                    continue
                # Positive directions visit each edge exactly once.
                for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if ni >= nx or nj >= ny or nk >= nz:
                        continue
                    d2 = distance_matrix[ni, nj, nk]
                    if (not voxel_matrix[ni, nj, nk] or not np.isfinite(d2) or d2 < 0):
                        continue
                    # Accept increasing or decreasing distances.
                    # Equal endpoint values do not define a unique crossing.
                    if d1 != d2 and min(d1, d2) <= dist <= max(d1, d2):
                        edges_containing_dist.add(((i, j, k), (ni, nj, nk)))

    # find the coordinate along the edge where the distance == dist
    intersection_points = []
    for (i1, j1, k1), (i2, j2, k2) in edges_containing_dist:
        d1 = distance_matrix[i1, j1, k1]
        d2 = distance_matrix[i2, j2, k2]
        t = (dist - d1) / (d2 - d1)
        intersection_point = (
            i1 + t * (i2 - i1),
            j1 + t * (j2 - j1),
            k1 + t * (k2 - k1)
        )
        intersection_points.append(intersection_point)

    # split disconected regions of intersection points into separate lists for plotting
    print("splitting disconnected regions...")
    disconnected_regions = split_regions(
        intersection_points,
        threshold=3.0,
    )

    slice_contours.append(disconnected_regions)

print("plotting...")
# plot a 3d surface plot of the intersection points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
import matplotlib.tri as mtri

max_triangle_edge = 3.0  # Voxel-index units, like your splitting threshold

for layer_regions in slice_contours:
    for region in layer_regions:
        points = np.unique(np.asarray(region, dtype=float), axis=0)
        if len(points) < 3:
            continue
        try:
            triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
        except (RuntimeError, ValueError):
            # XY projection may be degenerate for a vertical patch.
            ax.scatter(*points.T, s=1)
            continue
        triangles = points[triangulation.triangles]
        # Three physical edge lengths for every proposed triangle.
        edge_lengths = np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)

        reject = np.any(edge_lengths > max_triangle_edge, axis=1)

        if np.all(reject):
            ax.scatter(*points.T, s=1)
            continue

        triangulation.set_mask(reject)

        ax.plot_trisurf(
            triangulation,
            points[:, 2],
            linewidth=0.2,
            antialiased=True,
            alpha=1.0,
        )
ax.set_title(f"Distance slice: {dist} to {dist + dist_step}")
# set the axes limits to the voxel matrix shape
ax.set_xlim(0, distance_matrix.shape[0])
ax.set_ylim(0, distance_matrix.shape[1])
ax.set_zlim(0, distance_matrix.shape[2])
# set aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])
plt.show()
