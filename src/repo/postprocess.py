from __future__ import annotations

from collections import deque
from typing import Any

from .db import _db_path_from_repo, _sqlite_connect, load_repo_graph_db


def run_repo_postprocess(
    repo: dict[str, Any],
    *,
    flows: bool = True,
    communities: bool = True,
    fts: bool = True,
) -> dict[str, Any]:
    warnings: list[str] = []
    result: dict[str, Any] = {
        "repo_id": str(repo.get("id") or "").strip(),
        "fts_indexed": None,
        "flows_detected": None,
        "communities_detected": None,
        "summary": "",
    }

    db_path = _db_path_from_repo(repo)
    if not db_path.exists():
        return {
            "error": f"Repository DB not found: {db_path}",
            "warnings": warnings,
        }

    if fts:
        try:
            result["fts_indexed"] = _rebuild_repo_node_fts(repo)
        except Exception as exc:
            warnings.append(f"FTS rebuild failed: {exc}")
            result["fts_indexed"] = 0

    graph = load_repo_graph_db(repo)
    if flows:
        flows_list = _detect_repo_flows(graph)
        result["flows_detected"] = len(flows_list)
        result["flows"] = flows_list

    if communities:
        communities_list = _detect_repo_communities(graph)
        result["communities_detected"] = len(communities_list)
        result["communities"] = communities_list

    if warnings:
        result["warnings"] = warnings

    result["summary"] = (
        f"postprocess completed: {result.get('flows_detected', 0)} flows, "
        f"{result.get('communities_detected', 0)} communities, "
        f"fts_indexed={result.get('fts_indexed', 0)}"
    )
    return result


def _rebuild_repo_node_fts(repo: dict[str, Any]) -> int:
    db_path = _db_path_from_repo(repo)
    conn = _sqlite_connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS repo_nodes_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE repo_nodes_fts USING fts5("
            "name, path, rel_path, kind, symbol_kind, metadata"
            ")"
        )
        repo_id = str(repo.get("id") or "")
        count = 0
        for row in conn.execute(
            "SELECT name, path, rel_path, kind, symbol_kind, metadata "
            "FROM repo_nodes WHERE repo_id = ?",
            (repo_id,),
        ):
            conn.execute(
                "INSERT INTO repo_nodes_fts (name, path, rel_path, kind, symbol_kind, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(row[0] or ""),
                    str(row[1] or ""),
                    str(row[2] or ""),
                    str(row[3] or ""),
                    str(row[4] or ""),
                    str(row[5] or ""),
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def _detect_repo_flows(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = dict(graph.get("nodes") or {})
    edges = list(graph.get("edges") or [])

    file_nodes = {
        node_id: node for node_id, node in nodes.items() if str(node.get("kind") or "") == "file"
    }

    file_by_node: dict[str, str] = {}
    for node_id, node in nodes.items():
        kind = str(node.get("kind") or "")
        if kind == "file":
            file_by_node[node_id] = node_id
        elif kind == "symbol":
            rel_path = str(node.get("rel_path") or "").strip()
            if rel_path:
                file_by_node[node_id] = f"file:{rel_path}"

    adjacency: dict[str, set[str]] = {file_id: set() for file_id in file_nodes}
    in_degree: dict[str, int] = {file_id: 0 for file_id in file_nodes}
    for edge in edges:
        kind = str(edge.get("kind") or "")
        if kind not in {"imports", "calls", "references"}:
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        source_file = file_by_node.get(source)
        target_file = file_by_node.get(target)
        if source_file and target_file and source_file in adjacency and target_file in adjacency:
            adjacency[source_file].add(target_file)
            adjacency[target_file].add(source_file)
            in_degree[target_file] += 1

    entry_files = [fid for fid, degree in in_degree.items() if degree == 0]
    if not entry_files:
        entry_files = list(file_nodes)

    flows: list[dict[str, Any]] = []
    for root in sorted(entry_files):
        reachable = _bfs_file_reachability(root, adjacency)
        flow_files = sorted(reachable)
        flows.append(
            {
                "id": f"flow:{root}",
                "name": str(file_nodes[root].get("name") or root),
                "entry_file": root,
                "file_count": len(flow_files),
                "files": flow_files,
                "criticality": len(flow_files),
            }
        )

    return sorted(flows, key=lambda item: (-item["file_count"], item["entry_file"]))[:50]


def _bfs_file_reachability(root: str, adjacency: dict[str, set[str]]) -> set[str]:
    seen = {root}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _detect_repo_communities(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = dict(graph.get("nodes") or {})
    edges = list(graph.get("edges") or [])

    file_nodes = {
        node_id: node for node_id, node in nodes.items() if str(node.get("kind") or "") == "file"
    }

    file_by_node: dict[str, str] = {}
    for node_id, node in nodes.items():
        kind = str(node.get("kind") or "")
        if kind == "file":
            file_by_node[node_id] = node_id
        elif kind == "symbol":
            rel_path = str(node.get("rel_path") or "").strip()
            if rel_path:
                file_by_node[node_id] = f"file:{rel_path}"

    adjacency: dict[str, set[str]] = {file_id: set() for file_id in file_nodes}
    for edge in edges:
        kind = str(edge.get("kind") or "")
        if kind not in {"imports", "calls", "references"}:
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        source_file = file_by_node.get(source)
        target_file = file_by_node.get(target)
        if source_file and target_file and source_file in adjacency and target_file in adjacency:
            adjacency[source_file].add(target_file)
            adjacency[target_file].add(source_file)

    communities: list[dict[str, Any]] = []
    visited: set[str] = set()
    community_index = 1
    for file_id in sorted(file_nodes):
        if file_id in visited:
            continue
        group = _dfs_component(file_id, adjacency)
        visited.update(group)
        languages = sorted({str(file_nodes[fid].get("language") or "unknown") for fid in group})
        communities.append(
            {
                "id": f"community:{community_index}",
                "name": f"community-{community_index}",
                "size": len(group),
                "files": sorted(group),
                "languages": languages,
                "cohesion": sum(len(adjacency[fid]) for fid in group) / max(1, len(group)),
            }
        )
        community_index += 1

    return sorted(communities, key=lambda item: (-item["size"], item["id"]))


def _dfs_component(start: str, adjacency: dict[str, set[str]]) -> set[str]:
    stack = [start]
    seen: set[str] = {start}
    while stack:
        current = stack.pop()
        for neighbor in adjacency.get(current, []):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen
