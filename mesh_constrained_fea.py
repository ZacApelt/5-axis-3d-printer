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


def ammend_nodes(nodes):
    original_node_set = set(nodes)

    def orientation(first, second, third):
        return (
            (second.x - first.x) * (third.y - first.y)
            - (second.y - first.y) * (third.x - first.x)
        )

    def segments_cross(first, second, third, fourth):
        first_orientation = orientation(first, second, third)
        second_orientation = orientation(first, second, fourth)
        third_orientation = orientation(third, fourth, first)
        fourth_orientation = orientation(third, fourth, second)
        epsilon = 1e-9

        return (
            first_orientation * second_orientation < -epsilon
            and third_orientation * fourth_orientation < -epsilon
        )

    def connection_crosses(first, second, all_nodes):
        for third in all_nodes:
            for fourth in third.connections:
                if third in (first, second) or fourth in (first, second):
                    continue
                if segments_cross(first, second, third, fourth):
                    return True
        return False

    def connect(first, second):
        if first is second or connection_crosses(first, second, nodes):
            return False
        if second not in first.connections:
            first.connections.append(second)
        if first not in second.connections:
            second.connections.append(first)
        return True

    # Use a snapshot because connections are extended while new nodes are added.
    edge_nodes = [node for node in nodes if len(node.connections) < 6]

    connected_edge_pairs = []
    seen_pairs = set()
    for node in edge_nodes:
        for connected_node in node.connections:
            pair_key = frozenset((node, connected_node))
            if connected_node in edge_nodes and node is not connected_node and pair_key not in seen_pairs:
                connected_edge_pairs.append((node, connected_node))
                seen_pairs.add(pair_key)

    for node_a, node_b in connected_edge_pairs:
        n_b = np.array([node_b.x, node_b.y])
        n_a = np.array([node_a.x, node_a.y])
        ab = n_b - n_a
        length = np.linalg.norm(ab)
        if length < 1e-12 or length > 2 * hor_spacing:
            continue

        midpoint = (n_a + n_b) / 2
        normal = np.array([-ab[1], ab[0]]) / length
        candidate_height = np.sqrt(max(0.0, hor_spacing**2 - (length / 2) ** 2))
        candidate_points = (
            midpoint + candidate_height * normal,
            midpoint - candidate_height * normal,
        )

        for node_d in candidate_points:
            if not is_in_geometry(node_d[0], node_d[1]):
                continue

            too_close = any(
                np.linalg.norm(node_d - np.array([existing.x, existing.y]))
                < hor_spacing * 0.35
                for existing in nodes
            )
            if too_close:
                continue

            new_node = Node(node_d[0], node_d[1])
            nodes.append(new_node)
            connect(new_node, node_a)
            connect(new_node, node_b)

            # Only add local links that are short and do not cross an existing edge.
            candidate_nodes = set(node_a.connections + node_b.connections)
            candidate_nodes.discard(new_node)
            candidate_nodes.discard(node_a)
            candidate_nodes.discard(node_b)
            for connected_node in candidate_nodes:
                connected_position = np.array([connected_node.x, connected_node.y])
                if np.linalg.norm(node_d - connected_position) <= hor_spacing * 1.1:
                    connect(new_node, connected_node)

    # remove nodes that are outside the boundary of the original shape
    nodes_to_remove = {
        node for node in edge_nodes
        if not is_in_geometry(node.x, node.y)
    }
    if nodes_to_remove:
        nodes[:] = [node for node in nodes if node not in nodes_to_remove]
        remaining_nodes = set(nodes)
        for node in nodes:
            node.connections[:] = [
                connected_node
                for connected_node in node.connections
                if connected_node is not node and connected_node in remaining_nodes
            ]

    # Remove any duplicate links left by the original directed mesh and amendments.
    for node in nodes:
        node.connections[:] = list(dict.fromkeys(
            connected_node
            for connected_node in node.connections
            if connected_node is not node and connected_node in nodes
        ))

    # reapply forces to the nodes in the widened section
    # find the nodes that are vertically within vert_spacing of the udl line
    for node in nodes:
        if node.y >= mid_height - vert_spacing and node.y <= mid_height + vert_spacing:
            if node.x > minor_width:
                node.F = np.array([0, -1])  # add a downward force
            else:
                node.F = np.array([0, 0])  # remove any previous force
        else:
            node.F = np.array([0, 0])  # remove any previous force

    return set(nodes) != original_node_set


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

    for iteration in range(20):
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

    print(f"Amendment passes: {amendment_passes}")


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