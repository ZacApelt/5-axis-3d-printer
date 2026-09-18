import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y, fixed=False, connections=None):
        # x and y are the positions of the node in 2D space
        # fixed is a boolean indicating whether the node is fixed or not
        # connnections is a list of other nodes that this node is connected to
        self.x = x
        self.y = y
        self.fixed = fixed  # whether the node is fixed or not
        self.connections = [] if connections is None else connections  # list of other nodes that this node is connected to
        self.F = np.array([0.0, 0.0])  # force vector

class Spring:
    def __init__(self, node_a, node_b, stiffness, rest_length=None):
        self.node_a = node_a
        self.node_b = node_b
        self.stiffness = stiffness

        if rest_length is None:
            dx = node_b.x - node_a.x
            dy = node_b.y - node_a.y
            rest_length = np.hypot(dx, dy)

        self.rest_length = rest_length


# fill the shape with nodes following a tetrahedral mesh pattern
vert_spacing = 2
hor_spacing = vert_spacing *  2 / np.sqrt(3)

# define the shape with for loops rather than the shape array
nodes = []
rows = []
minor_width = 25
major_width = 2 * minor_width
height = 50
mid_height = height / 2

# Keep every layer exactly vert_spacing apart. The mesh is allowed to stop
# below the geometric height when the spacing does not divide it evenly.
y_positions = np.arange(0, height, vert_spacing)

for row_index, y in enumerate(y_positions):
    row = []
    width = major_width if y >= mid_height else minor_width
    for x in np.arange(0, width, hor_spacing):
        # every other row is offset by half the horizontal spacing
        if row_index % 2 == 0:
            row.append(Node(x, y))
        else:
            row.append(Node(x + hor_spacing / 2, y))
    rows.append(row)
    nodes.extend(row)

    if y >= mid_height and (row_index == 0 or y_positions[row_index - 1] < mid_height):
        # Apply the load to the first row in the widened section.
        for node in row:
            if node.x > minor_width:
                node.F = np.array([0, -1])  # add a downward force

# connect the nodes to their neighbors
for row_index, row in enumerate(rows):
    # Connect each node to the node on its right.
    for node, right_node in zip(row, row[1:]):
        node.connections.append(right_node)

    # Connect only to nodes in the immediately adjacent row.
    if row_index + 1 < len(rows):
        next_row = rows[row_index + 1]
        for node in row:
            for next_node in next_row:
                if abs(next_node.x - node.x) <= hor_spacing + 1e-9:
                    node.connections.append(next_node)

# make the first layer of nodes fixed
for node in rows[0]:
    node.fixed = True

print(f"Number of nodes: {len(nodes)}")

def is_in_geometry(x, y):
    if 0 <= x <= major_width and 0 <= y <= height:
        # within the bounding box, now check if it's within the shape
        if y < mid_height and x > minor_width:
            return False
        return True
    return False


# plot the nodes
'''
plt.figure(figsize=(8, 8))
for node in nodes:
    for conn in node.connections:
        plt.plot([node.x, conn.x], [node.y, conn.y], "k-")
    # plot the force vector as a red arrow

    if np.linalg.norm(node.F) > 0:
        print(node.F)
        plt.arrow(node.x, node.y, node.F[0] * 10, node.F[1] * 3, color="r", head_width=0.5)

    if node.fixed:
        plt.plot(node.x, node.y, "ro")
    else:
        plt.plot(node.x, node.y, "bo")
plt.show()
'''


def assemble_system(nodes, springs, force_scale=1.0):
    """
    Assemble the nonlinear internal force vector, external force vector,
    and tangent stiffness matrix at the current node positions.
    """
    node_count = len(nodes)
    dof_count = 2 * node_count

    node_indices = {node: i for i, node in enumerate(nodes)}

    internal_forces = np.zeros(dof_count)
    external_forces = np.zeros(dof_count)
    K = np.zeros((dof_count, dof_count))

    I2 = np.eye(2)

    # Assemble applied nodal forces
    for i, node in enumerate(nodes):
        external_forces[2 * i:2 * i + 2] = force_scale * node.F

    # Assemble spring forces and tangent stiffness
    for spring in springs:
        i = node_indices[spring.node_a]
        j = node_indices[spring.node_b]

        xi = np.array([
            spring.node_a.x,
            spring.node_a.y
        ])

        xj = np.array([
            spring.node_b.x,
            spring.node_b.y
        ])

        displacement = xj - xi
        current_length = np.linalg.norm(displacement)

        if current_length < 1e-12:
            raise ValueError(
                f"Spring between nodes {i} and {j} has zero length."
            )

        direction = displacement / current_length

        k = spring.stiffness
        rest_length = spring.rest_length

        # Positive when the spring is in tension
        tension = k * (current_length - rest_length)

        spring_force = tension * direction

        i_slice = slice(2 * i, 2 * i + 2)
        j_slice = slice(2 * j, 2 * j + 2)

        # Internal energy-gradient forces
        internal_forces[i_slice] -= spring_force
        internal_forces[j_slice] += spring_force

        direction_matrix = np.outer(direction, direction)

        # Tangent stiffness:
        # axial stiffness + geometric stiffness
        A = (
            k * direction_matrix
            + (tension / current_length)
            * (I2 - direction_matrix)
        )

        K[i_slice, i_slice] += A
        K[i_slice, j_slice] -= A
        K[j_slice, i_slice] -= A
        K[j_slice, j_slice] += A

    return K, internal_forces, external_forces


def solve_single_step(nodes, springs, force_scale=1.0, relaxation=1.0, max_node_displacement=None):
    """
    Perform exactly one linearised Newton step.

    This function does not iterate to equilibrium and does not change the
    applied-force scale. The caller controls both of those operations.

    Returns a dictionary containing convergence information.
    """
    K, internal_forces, external_forces = assemble_system(
        nodes,
        springs,
        force_scale
    )

    residual = external_forces - internal_forces

    fixed_dofs = []
    free_dofs = []

    for i, node in enumerate(nodes):
        node_dofs = [2 * i, 2 * i + 1]

        if node.fixed:
            fixed_dofs.extend(node_dofs)
        else:
            free_dofs.extend(node_dofs)

    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    free_dofs = np.asarray(free_dofs, dtype=int)

    if len(free_dofs) == 0:
        return {
            "residual_norm": 0.0,
            "displacement_norm": 0.0,
            "maximum_node_displacement": 0.0,
            "converged": True,
        }

    K_free = K[np.ix_(free_dofs, free_dofs)]
    residual_free = residual[free_dofs]

    try:
        delta_free = np.linalg.solve(K_free, residual_free)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "The stiffness matrix is singular. The mesh may contain an "
            "unconstrained rigid-body motion or a shear/floppy mode."
        ) from error

    delta_free *= relaxation

    delta = np.zeros(2 * len(nodes))
    delta[free_dofs] = delta_free
    delta = delta.reshape((-1, 2))

    maximum_displacement = np.max(
        np.linalg.norm(delta, axis=1)
    )

    # Optional protection against an excessively large Newton step
    if (
        max_node_displacement is not None
        and maximum_displacement > max_node_displacement
    ):
        reduction = max_node_displacement / maximum_displacement
        delta *= reduction
        delta_free *= reduction
        maximum_displacement = max_node_displacement

    # Update free-node positions
    for i, node in enumerate(nodes):
        if not node.fixed:
            node.x += delta[i, 0]
            node.y += delta[i, 1]

    residual_norm = np.linalg.norm(residual_free)
    force_norm = np.linalg.norm(external_forces[free_dofs])

    # Relative residual, with protection for zero applied force
    relative_residual = residual_norm / max(force_norm, 1.0)

    return {
        "residual_norm": residual_norm,
        "relative_residual": relative_residual,
        "displacement_norm": np.linalg.norm(delta_free),
        "maximum_node_displacement": maximum_displacement,
        "converged": relative_residual < 1e-8,
    }

def segments_cross(node_a, node_b, node_c, node_d, tolerance=1e-9):
    """
    Return True if AB and CD properly cross.

    Connections sharing a node are allowed.
    """
    if (
        node_a is node_c
        or node_a is node_d
        or node_b is node_c
        or node_b is node_d
    ):
        return False

    a = np.array([node_a.x, node_a.y])
    b = np.array([node_b.x, node_b.y])
    c = np.array([node_c.x, node_c.y])
    d = np.array([node_d.x, node_d.y])

    def orientation(p, q, r):
        return (
            (q[0] - p[0]) * (r[1] - p[1])
            - (q[1] - p[1]) * (r[0] - p[0])
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    return (
        (
            (o1 > tolerance and o2 < -tolerance)
            or (o1 < -tolerance and o2 > tolerance)
        )
        and
        (
            (o3 > tolerance and o4 < -tolerance)
            or (o3 < -tolerance and o4 > tolerance)
        )
    )


def connection_would_cross(node_a, node_b, nodes):
    """Check a proposed connection against every existing connection."""
    checked_edges = set()

    for existing_a in nodes:
        for existing_b in existing_a.connections:
            edge_key = frozenset((existing_a, existing_b))

            if edge_key in checked_edges:
                continue

            checked_edges.add(edge_key)

            if segments_cross(
                node_a,
                node_b,
                existing_a,
                existing_b,
            ):
                return True

    return False

def close_complete_hexagons(nodes, connect):
    """
    For every node with exactly six neighbours, sort those neighbours
    angularly and close any missing edges around the six-node ring.

    Returns True if at least one connection was added.
    """
    changed = False
    maximum_ring_edge_length = 1.5 * hor_spacing

    for centre_node in list(nodes):
        if len(centre_node.connections) != 6:
            continue

        centre_position = np.array([
            centre_node.x,
            centre_node.y,
        ])

        # Connections are not stored in geometric order, so explicitly
        # sort the six neighbours by their polar angle around the centre.
        ordered_neighbours = sorted(
            centre_node.connections,
            key=lambda neighbour: np.arctan2(
                neighbour.y - centre_node.y,
                neighbour.x - centre_node.x,
            ),
        )

        # Consecutive angular neighbours, including the final-to-first
        # pair, should form the six edges of the hexagonal ring.
        for index in range(6):
            first = ordered_neighbours[index]
            second = ordered_neighbours[(index + 1) % 6]

            if second in first.connections:
                continue

            distance = np.hypot(
                second.x - first.x,
                second.y - first.y,
            )

            # A large angular gap may represent a real external boundary,
            # not a missing hexagon edge.
            if distance > maximum_ring_edge_length:
                continue

            if connect(first, second):
                changed = True

    return changed

def opposite_triangle_candidates(
    node_a,
    node_b,
    candidate_positions,
):
    """
    If A-B already has one triangular face, return only the candidate
    on the opposite side. If it has no face, return both candidates.
    If it already has faces on both sides, return no candidates.
    """
    position_a = np.array([node_a.x, node_a.y])
    position_b = np.array([node_b.x, node_b.y])
    ab = position_b - position_a

    def side_of_edge(position):
        relative = position - position_a

        return (
            ab[0] * relative[1]
            - ab[1] * relative[0]
        )

    common_neighbours = (
        set(node_a.connections)
        & set(node_b.connections)
    )

    occupied_signs = []

    for common_node in common_neighbours:
        common_position = np.array([
            common_node.x,
            common_node.y,
        ])

        side = side_of_edge(common_position)

        if side > 1e-9:
            occupied_signs.append(1)
        elif side < -1e-9:
            occupied_signs.append(-1)

    occupied_signs = set(occupied_signs)

    # Existing triangular faces on both sides: the edge is internal and
    # no further triangle should be created.
    if occupied_signs == {-1, 1}:
        return []

    # No existing face: either candidate may represent vacant material.
    if not occupied_signs:
        return candidate_positions

    occupied_sign = next(iter(occupied_signs))

    return [
        candidate_position
        for candidate_position in candidate_positions
        if np.sign(side_of_edge(candidate_position))
        == -occupied_sign
    ]


def ammend_nodes(nodes):
    L = float(hor_spacing)
    eps = 1e-9 * L
    area_eps = eps * L
    merge_distance = 0.45 * L
    changed = False

    def position(node):
        return np.array([node.x, node.y], dtype=float)

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def on_segment(p, a, b):
        return (
            abs(cross(b - a, p - a)) <= area_eps
            and np.dot(p - a, p - b) <= eps**2
        )

    # Remove outside nodes before deciding where the boundary is.
    kept = [
        n for n in nodes
        if n.fixed or is_in_geometry(n.x, n.y)
    ]
    changed |= len(kept) != len(nodes)
    nodes[:] = kept
    live = set(nodes)

    # Use sets internally: symmetric, unique connections.
    adjacency = {n: set() for n in nodes}

    for a in nodes:
        for b in a.connections:
            if b in live and b is not a:
                adjacency[a].add(b)
                adjacency[b].add(a)

    if any(len(adjacency[n]) > 6 for n in nodes):
        raise RuntimeError(
            "The input mesh already has a node with >6 neighbours. "
            "Restart from the original mesh."
        )

    def all_edges():
        seen = set()
        result = []

        for a in nodes:
            for b in adjacency[a]:
                key = frozenset((a, b))
                if key not in seen:
                    seen.add(key)
                    result.append((a, b))

        return result

    def edges_conflict(a, b, c, d):
        # Identical edge is allowed.
        if frozenset((a, b)) == frozenset((c, d)):
            return False

        pa, pb = position(a), position(b)
        pc, pd = position(c), position(d)

        shared = {a, b} & {c, d}

        if shared:
            # Shared endpoints are allowed, but overlapping edges are not.
            joint = next(iter(shared))
            other_ab = b if a is joint else a
            other_cd = d if c is joint else c
            origin = position(joint)
            u = position(other_ab) - origin
            v = position(other_cd) - origin

            return (
                abs(cross(u, v)) <= area_eps
                and np.dot(u, v) > eps**2
            )

        s1 = cross(pb - pa, pc - pa)
        s2 = cross(pb - pa, pd - pa)
        s3 = cross(pd - pc, pa - pc)
        s4 = cross(pd - pc, pb - pc)

        def opposite(x, y):
            return (
                (x > area_eps and y < -area_eps)
                or (x < -area_eps and y > area_eps)
            )

        if opposite(s1, s2) and opposite(s3, s4):
            return True

        # Also reject touching a non-shared endpoint or collinear overlap.
        return (
            on_segment(pc, pa, pb)
            or on_segment(pd, pa, pb)
            or on_segment(pa, pc, pd)
            or on_segment(pb, pc, pd)
        )

    def inside_polygon(p, polygon):
        # Boundary counts as occupied.
        inside = False

        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            if on_segment(p, a, b):
                return True

            if (a[1] > p[1]) != (b[1] > p[1]):
                x_intersection = (
                    a[0]
                    + (p[1] - a[1])
                    * (b[0] - a[0])
                    / (b[1] - a[1])
                )
                if p[0] < x_intersection:
                    inside = not inside

        return inside

    def mesh_faces():
        """
        Walk directed edges with the face on the left.

        Positive-area walks are bounded cells.
        Negative-area walks border the exterior.
        """
        ordered = {
            n: sorted(
                adjacency[n],
                key=lambda other: np.arctan2(
                    other.y - n.y, other.x - n.x
                ),
            )
            for n in nodes
        }

        visited = set()
        cells = []
        boundary = []

        for a in nodes:
            for b in ordered[a]:
                if (a, b) in visited:
                    continue

                start = (a, b)
                edge = start
                walk = []

                while edge not in visited:
                    visited.add(edge)
                    walk.append(edge)
                    u, v = edge

                    neighbours = ordered[v]
                    incoming = neighbours.index(u)

                    # Clockwise turn from the reverse incoming edge.
                    w = neighbours[(incoming - 1) % len(neighbours)]
                    edge = (v, w)

                if edge != start:
                    raise RuntimeError("Invalid mesh face traversal.")

                polygon = [position(u) for u, _ in walk]
                twice_area = sum(
                    cross(p, q)
                    for p, q in zip(
                        polygon, polygon[1:] + polygon[:1]
                    )
                )

                if twice_area > area_eps:
                    cells.append(polygon)
                elif twice_area < -area_eps:
                    boundary.extend(walk)

        return cells, boundary

    def segment_in_geometry(a, b):
        """
        Exact segment check for your axis-aligned, L-shaped geometry:
        test between each crossing of a contour coordinate.
        """
        p, q = position(a), position(b)
        delta = q - p
        parameters = [0.0, 1.0]

        for axis, boundaries in (
            (0, (0.0, minor_width, major_width)),
            (1, (0.0, mid_height, height)),
        ):
            if abs(delta[axis]) > eps:
                for value in boundaries:
                    t = (value - p[axis]) / delta[axis]
                    if 0.0 < t < 1.0:
                        parameters.append(t)

        parameters = sorted(set(parameters))
        samples = parameters + [
            0.5 * (t0 + t1)
            for t0, t1 in zip(parameters, parameters[1:])
        ]

        return all(
            is_in_geometry(*(p + t * delta))
            for t in samples
        )

    edges = all_edges()

    # Face walking requires an initially planar mesh.
    for index, (a, b) in enumerate(edges):
        for c, d in edges[index + 1:]:
            if edges_conflict(a, b, c, d):
                raise RuntimeError(
                    "The input mesh already has crossing/overlapping "
                    "springs. Restart from the original mesh."
                )

    cells, boundary = mesh_faces()

    # Snapshot: newly exposed edges are handled by your next amendment pass.
    seeds = sorted(
        boundary,
        key=lambda edge: np.linalg.norm(
            position(edge[1]) - position(edge[0])
        ),
    )
    exposed = set(boundary)

    for a, b in seeds:
        # Earlier insertions may already have covered this edge.
        if (a, b) not in exposed:
            continue

        pa, pb = position(a), position(b)
        ab = pb - pa
        length = np.linalg.norm(ab)

        # Two nominal-length sides cannot span an edge >= 2L.
        if length <= eps or length >= 2.0 * L - eps:
            continue

        normal = np.array([-ab[1], ab[0]]) / length
        altitude = np.sqrt(L**2 - 0.25 * length**2)

        # Exterior face is on the left of this directed boundary edge.
        target = 0.5 * (pa + pb) + altitude * normal

        # First try reusing a nearby node; otherwise propose a new one.
        candidates = sorted(
            (
                n for n in nodes
                if n is not a and n is not b
                and np.linalg.norm(position(n) - target)
                < merge_distance
                and cross(ab, position(n) - pa) > area_eps
            ),
            key=lambda n: np.linalg.norm(position(n) - target),
        )

        if (
            is_in_geometry(*target)
            and all(
                np.linalg.norm(position(n) - target) >= merge_distance
                for n in nodes
            )
            and not any(inside_polygon(target, cell) for cell in cells)
        ):
            candidates.append(Node(float(target[0]), float(target[1])))

        for c in candidates:
            pc = position(c)
            is_new = c not in adjacency

            if cross(ab, pc - pa) <= area_eps:
                continue

            if not is_in_geometry(*pc):
                continue

            # A triangle cannot enclose another mesh node, including on
            # one of its sides. This prevents covering existing material
            # or adding an edge through an intermediate node.
            contains_node = False

            for n in nodes:
                if n is a or n is b or n is c:
                    continue

                p = position(n)
                sides = (
                    cross(pb - pa, p - pa),
                    cross(pc - pb, p - pb),
                    cross(pa - pc, p - pc),
                )

                if min(sides) >= -area_eps:
                    contains_node = True
                    break

            if contains_node:
                continue

            missing = [
                (u, c) for u in (a, b)
                if c not in adjacency[u]
            ]
            if not missing:
                continue

            increases = {a: 0, b: 0, c: 0}
            for u, v in missing:
                increases[u] += 1
                increases[v] += 1

            if any(
                len(adjacency.get(n, ())) + increase > 6
                for n, increase in increases.items()
            ):
                continue

            if any(
                not segment_in_geometry(u, v)
                for u, v in missing
            ):
                continue

            if any(
                edges_conflict(u, v, e, f)
                for u, v in missing
                for e, f in edges
            ):
                continue

            # Accept the entire triangle together.
            if is_new:
                nodes.append(c)
                adjacency[c] = set()

            for u, v in missing:
                adjacency[u].add(v)
                adjacency[v].add(u)
                edges.append((u, v))

            changed = True

            # Recompute occupancy immediately, not next iteration.
            cells, boundary = mesh_faces()
            exposed = set(boundary)
            break

    # Write the authoritative undirected graph back to your Node objects.
    index = {n: i for i, n in enumerate(nodes)}

    for n in nodes:
        neighbours = sorted(adjacency[n], key=index.get)

        if (
            set(n.connections) != adjacency[n]
            or len(n.connections) != len(neighbours)
        ):
            changed = True

        n.connections[:] = neighbours

        # Preserve the load assignment from your latest uploaded script.
        if (
            mid_height - vert_spacing <= n.y <= mid_height + vert_spacing
            and n.x > minor_width
        ):
            n.F = np.array([0.0, -1.0])
        else:
            n.F = np.zeros(2)

    return changed


def build_springs(nodes):
    springs = []
    seen_pairs = set()
    for node in nodes:
        for connected_node in node.connections:
            if node is connected_node:
                continue
            pair_key = frozenset((node, connected_node))
            if pair_key not in seen_pairs:
                springs.append(Spring(node, connected_node, spring_stiffness))
                seen_pairs.add(pair_key)
    return springs


# Create springs from the initial undeformed geometry
spring_stiffness = 5.0

springs = build_springs(nodes)

# Save original positions for plotting
original_positions = {
    node: np.array([node.x, node.y])
    for node in nodes
}

# Gradually apply the node.F loads
number_of_load_steps = 20

for load_step in range(1, number_of_load_steps + 1 , 1):
    force_scale = load_step / number_of_load_steps

    for iteration in range(50):
        result = solve_single_step(
            nodes,
            springs,
            force_scale=force_scale,
            relaxation=0.5,
            max_node_displacement=0.5,
        )

        if (result["relative_residual"] < 1e-7 and result["maximum_node_displacement"] < 1e-7):
            print(f"Converged at load step {load_step}, iteration {iteration + 1}")
            break

    print(
        f"Load {force_scale:.2f}: "
        f"iterations={iteration + 1}, "
        f"residual={result['relative_residual']:.3e}"
    )

    # Repeat amendment passes until the mesh reaches a stable node set.
    amendment_passes = 0
    max_amendment_passes = 20
    while True:
        amendment_passes += 1
        nodes_changed = ammend_nodes(nodes)
        springs = build_springs(nodes)
        for node in nodes:
            if node not in original_positions:
                original_positions[node] = np.array([node.x, node.y])

        if not nodes_changed:
            break
        if amendment_passes >= max_amendment_passes:
            raise RuntimeError(
                "Node amendment did not stabilize within "
                f"{max_amendment_passes} passes."
            )

    #print(f"Amendment passes: {amendment_passes}")


        plt.figure(figsize=(8, 8))

        # plot the geometry boundary
        plt.plot([0, minor_width, minor_width, major_width, major_width, 0, 0], [0, 0, mid_height, mid_height, height, height, 0], "k-", linewidth=2)

        # Deformed mesh
        for spring in springs:
            plt.plot(
                [spring.node_a.x, spring.node_b.x],
                [spring.node_a.y, spring.node_b.y],
                "k-",
                linewidth=1,
            )

        for node in nodes:
            colour = "red" if node.fixed else "blue"
            plt.plot(node.x, node.y, "o", color=colour)

            # plot the force vector as a red arrow
            if np.linalg.norm(node.F) > 0:
                plt.arrow(node.x, node.y, node.F[0] * 3, node.F[1] * 3, color="r", head_width=0.5)

        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()