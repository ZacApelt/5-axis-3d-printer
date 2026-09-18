import numpy as np
import trimesh
import matplotlib.pyplot as plt
import heapq
from itertools import product


mesh = trimesh.load_mesh("5-axis-3d-printer\\stl\\benchy.stl")

print("stl loaded")

voxel_size = 0.5
voxel_mesh = mesh.voxelized(pitch=voxel_size)

# fill the interior of the mesh to get a solid voxel representation
voxel_mesh.fill()

print("voxelised")

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


distance_matrix = voxel_distances(voxel_matrix, pitch=voxel_size)

# set all inf values to -1
distance_matrix[distance_matrix == np.inf] = -1

slice_thickenss = 2
for i in np.arange(0, distance_matrix.max(), slice_thickenss):
    # search the distance matrix for i < distance < i + slice_thickenss
    slice_mask = (distance_matrix >= i) & (distance_matrix < i + slice_thickenss)
    # plot a 3d scatter of the slice
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*np.where(slice_mask), c='b', marker='o', s=1)
    ax.set_title(f"Distance slice: {i} to {i + slice_thickenss}")
    # set the axes limits to the voxel matrix shape
    ax.set_xlim(0, voxel_matrix.shape[0])
    ax.set_ylim(0, voxel_matrix.shape[1])
    ax.set_zlim(0, voxel_matrix.shape[2])
    plt.show()





# display the voxelised mesh
# scene = trimesh.Scene(voxel_mesh)
# scene.show()