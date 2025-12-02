from typing import Dict, Iterable, List, Tuple

# Dict of node name mapped to list of (neighbor node name, edge label)
NeighborMap = Dict[str, List[Tuple[str, str]]]


def build_edge_neighbor_map(node_names: Iterable[str], topology: str) -> NeighborMap:
    """
    Build a neighbor map describing which nodes connect together and
    which edge labels to assign for a particular topology.
    """
    normalized = str(topology).lower()
    node_set = set(node_names)

    if normalized == "full":
        return _build_complete_topology_map(node_set)
    if normalized == "neighbor_1":
        return _build_hex_topology_map(node_set, include_second_order=False)
    if normalized == "neighbor_2":
        return _build_hex_topology_map(node_set, include_second_order=True)

    raise ValueError(
        f"Unsupported edge topology '{topology}'. "
        "Supported values: 'full', 'neighbor_1', 'neighbor_2'."
    )


def _build_complete_topology_map(node_set: Iterable[str]) -> NeighborMap:
    neighbor_map: NeighborMap = {}
    for node in node_set:
        neighbor_map[node] = [
            (other, "Plain") for other in node_set if other != node
        ]
    return neighbor_map


def _parse_node_coordinates(node_name: str) -> Tuple[int, int]:
    row_str, col_str = node_name.split(":")
    return int(row_str), int(col_str)


def _build_hex_topology_map(
    node_set: Iterable[str],
    *,
    include_second_order: bool,
) -> NeighborMap:
    neighbor_map: NeighborMap = {node: [] for node in node_set}

    coords = [_parse_node_coordinates(node) for node in node_set]
    max_row = max(row for row, _ in coords)
    max_col = max(col for _, col in coords)

    direction_offsets = [
        ("TopLeft", (-1, 0)),
        ("TopRight", (-1, 1)),
        ("Left", (0, -1)),
        ("Right", (0, 1)),
        ("BottomLeft", (1, -1)),
        ("BottomRight", (1, 0)),
    ]

    second_order_offsets = [
        ("SecondTop", (-2, 0)),
        ("SecondTopRight", (-2, 1)),
        ("SecondUpperLeft", (-1, -1)),
        ("SecondTopFarRight", (-2, 2)),
        ("SecondUpperFarRight", (-1, 2)),
        ("SecondLeftFar", (0, -2)),
        ("SecondLowerLeft", (1, -2)),
        ("SecondRightFar", (0, 2)),
        ("SecondLowerRight", (1, 1)),
        ("SecondBottomFarLeft", (2, -2)),
        ("SecondBottomLeft", (2, -1)),
        ("SecondBottom", (2, 0)),
    ]

    for node in node_set:
        row, col = _parse_node_coordinates(node)
        neighbors: List[Tuple[str, str]] = []
        for label, (dr, dc) in direction_offsets:
            nr, nc = row + dr, col + dc
            if 1 <= nr <= max_row and 1 <= nc <= max_col:
                neighbor_name = f"{nr}:{nc}"
                if neighbor_name in node_set:
                    neighbors.append((neighbor_name, label))

        if include_second_order:
            for label, (dr, dc) in second_order_offsets:
                nr, nc = row + dr, col + dc
                if 1 <= nr <= max_row and 1 <= nc <= max_col:
                    neighbor_name = f"{nr}:{nc}"
                    if neighbor_name in node_set:
                        neighbors.append((neighbor_name, label))

        neighbor_map[node] = neighbors

    return neighbor_map
