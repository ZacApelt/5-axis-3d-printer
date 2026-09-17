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


# fill the shape with nodes following a tetrahedral mesh pattern
vert_spacing = 5
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

# plot the nodes
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