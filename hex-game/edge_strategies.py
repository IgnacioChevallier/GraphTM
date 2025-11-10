from typing import Dict, Iterable, List, Tuple

NeighborMap = Dict[str, List[Tuple[str, str]]]


def build_edge_neighbor_map(node_names: Iterable[str], topology: str) -> NeighborMap:
    topology = str(topology).lower()
    node_set = set(node_names)

    if topology == "complete":
        return _build_complete_topology_map(node_set)
    if topology == "hex":
        return _build_hex_topology_map(node_set)
    if topology == "square":
        return _build_square_topology_map(node_set)
    if topology == "triangle":
        return _build_triangle_topology_map(node_set)

    raise ValueError(
        f"Unsupported edge_topology '{topology}'. Supported values: 'hex', 'complete'."
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


def _build_hex_topology_map(node_set: Iterable[str]) -> NeighborMap:
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

    for node in node_set:
        row, col = _parse_node_coordinates(node)
        neighbors: List[Tuple[str, str]] = []
        for label, (dr, dc) in direction_offsets:
            nr, nc = row + dr, col + dc
            if nr < 1 or nr > max_row or nc < 1 or nc > max_col:
                continue
            neighbor_name = f"{nr}:{nc}"
            if neighbor_name in node_set:
                neighbors.append((neighbor_name, label))
        neighbor_map[node] = neighbors

    return neighbor_map


def _build_square_topology_map(node_set: Iterable[str]) -> NeighborMap:
    # TODO: Implement square topology map
    return None

def _build_triangle_topology_map(node_set: Iterable[str]) -> NeighborMap:
    # TODO: Implement triangle topology map
    return None