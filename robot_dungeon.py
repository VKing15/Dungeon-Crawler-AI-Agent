#!/usr/bin/env python3
"""
Robot Dungeon – Vikyrthan Kirushnamoorthy

Robot Dungeon is a 2D virtual world designed to demonstrate an LLM-driven agent harness.

The environment is partially observable: the robot can only perceive the room or
corridor segment it currently occupies. The agent maintains memory of explored
rooms, doors, the key, and the exit, and uses a compact action space to navigate
and complete its objective.

The primary task is to explore the dungeon, locate the key, find the exit, and
escape successfully.

Gemini is optional and is used as a high-level planner for goal selection.
When Gemini is unavailable, the agent falls back to a built-in heuristic policy,
ensuring the system remains fully functional and reproducible.
"""

from __future__ import annotations

import json
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import random
import textwrap
import urllib.request
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import sys
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated as an API.*")
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

# Silence pygame's startup banner and any import-time warnings so the console
# only shows the submission log lines we care about.
_stdout_sink = StringIO()
_stderr_sink = StringIO()
with redirect_stdout(_stdout_sink), redirect_stderr(_stderr_sink):
    import pygame

Point = Tuple[int, int]
Area = Tuple[str, int]  # ("room"|"corridor", id)

TILE_SIZE = 32
GRID_W = 33
GRID_H = 25
ROOM_COUNT = 8
FPS = 60
ACTION_DELAY = 0.20
MAX_TURNS = 900

WALL = "#"
FLOOR = "."
DOOR = "+"
CORRIDOR = "="

DIRS: Dict[str, Point] = {
    "move_north": (0, -1),
    "move_south": (0, 1),
    "move_west": (-1, 0),
    "move_east": (1, 0),
}
DIR_FROM_DELTA = {v: k for k, v in DIRS.items()}

ROOM_PALETTE = [
    (196, 166, 126),
    (178, 121, 121),
    (120, 150, 195),
    (165, 132, 196),
    (128, 190, 132),
    (125, 188, 194),
    (214, 198, 109),
    (212, 132, 118),
    (154, 181, 120),
    (186, 143, 204),
]


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def darken(color: Tuple[int, int, int], amount: int = 55) -> Tuple[int, int, int]:
    r, g, b = color
    return (max(0, r - amount), max(0, g - amount), max(0, b - amount))


@dataclass(frozen=True)
class Room:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Point:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def contains_footprint(self, pos: Point) -> bool:
        x, y = pos
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def contains_interior(self, pos: Point) -> bool:
        x, y = pos
        return (self.x + 1) <= x < (self.x + self.w - 1) and (self.y + 1) <= y < (self.y + self.h - 1)

    def intersects(self, other: "Room", padding: int = 2) -> bool:
        return not (
            self.x + self.w + padding <= other.x
            or other.x + other.w + padding <= self.x
            or self.y + self.h + padding <= other.y
            or other.y + other.h + padding <= self.y
        )


@dataclass
class RoomMemory:
    room_id: int
    first_seen_turn: int
    tiles_seen: Set[Point] = field(default_factory=set)
    doors_seen: Set[Point] = field(default_factory=set)
    doors_used: Set[Point] = field(default_factory=set)
    door_attempt_counts: Dict[Point, int] = field(default_factory=dict)
    connected_rooms: Set[int] = field(default_factory=set)
    key_seen: Optional[Point] = None
    exit_seen: Optional[Point] = None
    explored: bool = False
    summary: str = ""


@dataclass
class CorridorMemory:
    corridor_id: int
    tiles_seen: Set[Point] = field(default_factory=set)
    doors_seen: Set[Point] = field(default_factory=set)
    doors_visited: Set[Point] = field(default_factory=set)
    door_attempt_counts: Dict[Point, int] = field(default_factory=dict)
    explored: bool = False


class Dungeon:
    def __init__(self, width: int = GRID_W, height: int = GRID_H, room_count: int = ROOM_COUNT):
        self.width = width
        self.height = height
        self.room_count = room_count

        self.grid = [[WALL for _ in range(width)] for _ in range(height)]
        self.room_ids = [[-1 for _ in range(width)] for _ in range(height)]
        self.corridor_ids = [[-1 for _ in range(width)] for _ in range(height)]

        self.rooms: List[Room] = []
        self.room_tiles: Dict[int, Set[Point]] = defaultdict(set)
        self.room_doors: Dict[int, Set[Point]] = defaultdict(set)
        self.corridor_tiles: Dict[int, Set[Point]] = defaultdict(set)
        self.corridor_doors: Dict[int, Set[Point]] = defaultdict(set)
        self.room_graph: Dict[int, Set[int]] = defaultdict(set)

        self.items: Dict[Point, str] = {}
        self.start: Point = (1, 1)
        self.key_pos: Point = (1, 1)
        self.exit_pos: Point = (width - 2, height - 2)

        self.generate()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def carve_tile(self, x: int, y: int, kind: str, room_id: int = -1, corridor_id: int = -1) -> None:
        if not self.in_bounds(x, y):
            return
        self.grid[y][x] = kind
        self.room_ids[y][x] = room_id
        self.corridor_ids[y][x] = corridor_id
        if room_id >= 0:
            self.room_tiles[room_id].add((x, y))
        if corridor_id >= 0:
            self.corridor_tiles[corridor_id].add((x, y))

    def carve_room(self, room: Room, room_id: int) -> None:
        for y in range(room.y, room.y + room.h):
            for x in range(room.x, room.x + room.w):
                if x in (room.x, room.x + room.w - 1) or y in (room.y, room.y + room.h - 1):
                    self.carve_tile(x, y, WALL, room_id=room_id)
                else:
                    self.carve_tile(x, y, FLOOR, room_id=room_id)

    def open_door(self, room: Room, door: Point, room_id: int) -> None:
        x, y = door
        if room.contains_footprint(door):
            self.carve_tile(x, y, DOOR, room_id=room_id)
            self.room_doors[room_id].add(door)

    def door_side(self, room: Room, door: Point) -> str:
        x, y = door
        if x == room.x:
            return "W"
        if x == room.x + room.w - 1:
            return "E"
        if y == room.y:
            return "N"
        if y == room.y + room.h - 1:
            return "S"
        return "?"

    def door_too_close(self, room_id: int, door: Point, min_distance: int = 2) -> bool:
        """Reject doors that are immediately adjacent or diagonally touching.

        This keeps individual gateways visually and logically distinct so the
        agent does not oscillate between two neighboring doors that are really
        the same choice in practice.
        """
        for other in self.room_doors.get(room_id, set()):
            if max(abs(other[0] - door[0]), abs(other[1] - door[1])) < min_distance:
                return True
        return False

    def carve_corridor(self, path: List[Point], corridor_id: int) -> None:
        for x, y in path:
            self.carve_tile(x, y, CORRIDOR, corridor_id=corridor_id)

    def walkable_for_corridor(self, pos: Point, room_exceptions: Set[Point]) -> bool:
        x, y = pos
        if not self.in_bounds(x, y):
            return False
        if pos in room_exceptions:
            return True
        return self.room_ids[y][x] < 0 and self.corridor_ids[y][x] < 0

    def bfs_path(self, start: Point, goal: Point, room_exceptions: Set[Point]) -> List[Point]:
        if start == goal:
            return [start]

        q: Deque[Point] = deque([start])
        came_from: Dict[Point, Optional[Point]] = {start: None}

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            cx, cy = cur
            for nxt in ((cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)):
                if nxt in came_from:
                    continue
                if nxt != goal and not self.walkable_for_corridor(nxt, room_exceptions):
                    continue
                came_from[nxt] = cur
                q.append(nxt)

        if goal not in came_from:
            return []

        path = [goal]
        cur = goal
        while came_from[cur] is not None:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    def room_at(self, pos: Point) -> Optional[int]:
        x, y = pos
        if not self.in_bounds(x, y):
            return None
        rid = self.room_ids[y][x]
        return None if rid < 0 else rid

    def corridor_at(self, pos: Point) -> Optional[int]:
        x, y = pos
        if not self.in_bounds(x, y):
            return None
        cid = self.corridor_ids[y][x]
        return None if cid < 0 else cid

    def area_at(self, pos: Point) -> Optional[Area]:
        cid = self.corridor_at(pos)
        if cid is not None:
            return ("corridor", cid)
        rid = self.room_at(pos)
        if rid is not None:
            return ("room", rid)
        return None

    def room_width(self, room: Room) -> int:
        return room.w

    def room_height(self, room: Room) -> int:
        return room.h

    def _door_side(self, room: Room, door: Point) -> Optional[str]:
        x, y = door
        if x == room.x:
            return "left"
        if x == room.x + room.w - 1:
            return "right"
        if y == room.y:
            return "top"
        if y == room.y + room.h - 1:
            return "bottom"
        return None

    def _door_conflicts_with_room(self, room_id: int, door: Point) -> bool:
        room = self.rooms[room_id]
        side = self._door_side(room, door)
        if side is None:
            return True

        # Keep doors on the same wall far enough apart that the agent never
        # sees them as a single toggling choice.
        for other in self.room_doors.get(room_id, set()):
            if other == door:
                continue
            if self._door_side(room, other) != side:
                continue
            if abs(other[0] - door[0]) + abs(other[1] - door[1]) <= 2:
                return True
        return False

    def _choose_doors_for_pair(self, a: Room, b: Room, rng: random.Random) -> Tuple[Point, Point, Point, Point]:
        ax, ay = a.center
        bx, by = b.center

        if abs(ax - bx) >= abs(ay - by):
            # Horizontal connection.
            a_y = clamp(by + rng.randint(-1, 1), a.y + 1, a.y + a.h - 2)
            b_y = clamp(a_y + rng.randint(-1, 1), b.y + 1, b.y + b.h - 2)
            if bx >= ax:
                door_a = (a.x + a.w - 1, a_y)
                door_b = (b.x, b_y)
                start = (door_a[0] + 1, door_a[1])
                end = (door_b[0] - 1, door_b[1])
            else:
                door_a = (a.x, a_y)
                door_b = (b.x + b.w - 1, b_y)
                start = (door_a[0] - 1, door_a[1])
                end = (door_b[0] + 1, door_b[1])
        else:
            # Vertical connection.
            a_x = clamp(bx + rng.randint(-1, 1), a.x + 1, a.x + a.w - 2)
            b_x = clamp(a_x + rng.randint(-1, 1), b.x + 1, b.x + b.w - 2)
            if by >= ay:
                door_a = (a_x, a.y + a.h - 1)
                door_b = (b_x, b.y)
                start = (door_a[0], door_a[1] + 1)
                end = (door_b[0], door_b[1] - 1)
            else:
                door_a = (a_x, a.y)
                door_b = (b_x, b.y + b.h - 1)
                start = (door_a[0], door_a[1] - 1)
                end = (door_b[0], door_b[1] + 1)

        return door_a, door_b, start, end

    def _path_avoids_rooms(self, path: List[Point], start: Point, end: Point) -> bool:
        for p in path:
            if p == start or p == end:
                continue
            x, y = p
            if self.room_ids[y][x] >= 0:
                return False
        return True

    def connect_rooms(self, idx_a: int, idx_b: int, rng: random.Random) -> bool:
        if idx_a == idx_b:
            return False

        if idx_b in self.room_graph.get(idx_a, set()):
            return False

        a = self.rooms[idx_a]
        b = self.rooms[idx_b]
        door_a, door_b, start, end = self._choose_doors_for_pair(a, b, rng)

        if not self.in_bounds(*start) or not self.in_bounds(*end):
            return False
        if self.room_at(start) is not None or self.room_at(end) is not None:
            return False
        if self.door_too_close(idx_a, door_a) or self.door_too_close(idx_b, door_b):
            return False

        def door_conflicts() -> bool:
            return self._door_conflicts_with_room(idx_a, door_a) or self._door_conflicts_with_room(idx_b, door_b)

        # Pathfind around rooms using empty space only.
        room_exceptions = {start, end}
        path = self.bfs_path(start, end, room_exceptions)
        if not path or not self._path_avoids_rooms(path, start, end) or door_conflicts():
            # One retry with the alternate axis.
            ax, ay = a.center
            bx, by = b.center
            if abs(ax - bx) >= abs(ay - by):
                # Try vertical style instead.
                a_x = clamp(bx, a.x + 1, a.x + a.w - 2)
                b_x = clamp(ax, b.x + 1, b.x + b.w - 2)
                if by >= ay:
                    door_a = (a_x, a.y + a.h - 1)
                    door_b = (b_x, b.y)
                    start = (door_a[0], door_a[1] + 1)
                    end = (door_b[0], door_b[1] - 1)
                else:
                    door_a = (a_x, a.y)
                    door_b = (b_x, b.y + b.h - 1)
                    start = (door_a[0], door_a[1] - 1)
                    end = (door_b[0], door_b[1] + 1)
            else:
                a_y = clamp(by, a.y + 1, a.y + a.h - 2)
                b_y = clamp(ay, b.y + 1, b.y + b.h - 2)
                if bx >= ax:
                    door_a = (a.x + a.w - 1, a_y)
                    door_b = (b.x, b_y)
                    start = (door_a[0] + 1, door_a[1])
                    end = (door_b[0] - 1, door_b[1])
                else:
                    door_a = (a.x, a_y)
                    door_b = (b.x + b.w - 1, b_y)
                    start = (door_a[0] - 1, door_a[1])
                    end = (door_b[0] + 1, door_b[1])

            room_exceptions = {start, end}
            path = self.bfs_path(start, end, room_exceptions)
            if not path or not self._path_avoids_rooms(path, start, end) or door_conflicts():
                return False
            if self.door_too_close(idx_a, door_a) or self.door_too_close(idx_b, door_b):
                return False

        corridor_id = len(self.corridor_tiles)
        self.carve_corridor(path, corridor_id)
        self.open_door(a, door_a, idx_a)
        self.open_door(b, door_b, idx_b)
        self.corridor_doors[corridor_id].update({door_a, door_b})
        self.room_graph[idx_a].add(idx_b)
        self.room_graph[idx_b].add(idx_a)
        return True

    def _distance_map_from_room(self, start_room: int) -> Dict[int, int]:
        q: Deque[int] = deque([start_room])
        dist = {start_room: 0}
        while q:
            cur = q.popleft()
            for nxt in self.room_graph.get(cur, set()):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist

    def generate(self) -> None:
        rng = random.Random()
        self.grid = [[WALL for _ in range(self.width)] for _ in range(self.height)]
        self.room_ids = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        self.corridor_ids = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        self.rooms.clear()
        self.room_tiles.clear()
        self.room_doors.clear()
        self.corridor_tiles.clear()
        self.corridor_doors.clear()
        self.room_graph.clear()
        self.items.clear()

        # Fixed room placement keeps the map readable, but the connections now
        # branch and loop so the robot has real choices instead of a single snake.
        layout = [
            (3, 1, 7, 5),
            (14, 1, 7, 5),
            (14, 7, 7, 5),
            (3, 7, 7, 5),
            (3, 13, 7, 5),
            (14, 13, 7, 5),
            (14, 19, 7, 5),
            (3, 19, 7, 5),
        ]

        for rid, (x, y, w, h) in enumerate(layout):
            room = Room(x, y, w, h)
            self.rooms.append(room)
            self.carve_room(room, rid)

        def degree(room_id: int) -> int:
            return len(self.room_graph.get(room_id, set()))

        def linked(a: int, b: int) -> bool:
            return b in self.room_graph.get(a, set())

        def try_link(a: int, b: int) -> bool:
            if a == b or linked(a, b):
                return False
            if degree(a) >= 4 or degree(b) >= 4:
                return False
            return self.connect_rooms(a, b, rng)

        # Candidate links ordered by layout proximity. These are all compatible
        # with the fixed room placement, and the random order makes each run
        # slightly different.
        candidate_pairs = [
            (0, 1), (0, 3),
            (1, 2), (1, 5),
            (2, 3), (2, 5), (2, 6),
            (3, 4), (3, 5),
            (4, 5), (4, 7),
            (5, 6), (5, 7),
            (6, 7),
        ]
        rng.shuffle(candidate_pairs)

        # First, make sure the graph is connected by growing from room 0.
        connected = {0}
        remaining = set(range(1, len(self.rooms)))
        guard = 0
        while remaining and guard < 200:
            guard += 1
            progress = False
            rng.shuffle(candidate_pairs)
            for a, b in candidate_pairs:
                if a in connected and b in remaining:
                    if try_link(a, b):
                        connected.add(b)
                        remaining.remove(b)
                        progress = True
                        break
                elif b in connected and a in remaining:
                    if try_link(b, a):
                        connected.add(a)
                        remaining.remove(a)
                        progress = True
                        break
            if not progress:
                break

        # Fallback: if anything could not be attached, use the original snake
        # backbone so the dungeon always remains playable.
        if remaining:
            fallback_chain = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
            for a, b in fallback_chain:
                try_link(a, b)
            connected = set(range(len(self.rooms)))
            remaining.clear()

        # Add extra links to create loops and alternate routes.
        extra_links = rng.randint(3, 5)
        extra_candidates = candidate_pairs[:]
        rng.shuffle(extra_candidates)
        for a, b in extra_candidates:
            if extra_links <= 0:
                break
            if try_link(a, b):
                extra_links -= 1

        # Key and exit are chosen by graph distance, not by room order.
        self.start = self.rooms[0].center
        dist_from_start = self._distance_map_from_room(0)
        if len(dist_from_start) < len(self.rooms):
            # Absolute safety net.
            for idx in range(len(self.rooms) - 1):
                try_link(idx, idx + 1)
            dist_from_start = self._distance_map_from_room(0)

        farthest_distance = max(dist_from_start.get(rid, 0) for rid in range(len(self.rooms)))
        exit_candidates = [rid for rid in range(len(self.rooms)) if dist_from_start.get(rid, -1) == farthest_distance]
        self.exit_room = rng.choice(exit_candidates)
        self.exit_pos = self.rooms[self.exit_room].center

        dist_from_exit = self._distance_map_from_room(self.exit_room)
        key_candidates = [
            rid
            for rid in range(len(self.rooms))
            if rid not in {0, self.exit_room} and dist_from_exit.get(rid, -1) >= 2
        ]
        if not key_candidates:
            key_candidates = [rid for rid in range(len(self.rooms)) if rid not in {0, self.exit_room}]
        key_candidates.sort(
            key=lambda rid: (
                dist_from_exit.get(rid, -1),
                dist_from_start.get(rid, -1),
                rng.random(),
            ),
            reverse=True,
        )
        self.key_room = key_candidates[0]
        self.key_pos = self.rooms[self.key_room].center
        if self.key_pos == self.exit_pos:
            for rid in key_candidates[1:]:
                if self.rooms[rid].center != self.exit_pos:
                    self.key_room = rid
                    self.key_pos = self.rooms[rid].center
                    break

        self.items[self.key_pos] = "key"

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.grid[y][x] in {FLOOR, DOOR, CORRIDOR}

    def visible_tiles(self, robot: Point) -> Set[Point]:
        area = self.area_at(robot)
        visible: Set[Point] = set()
        if area is None:
            return visible

        kind, idx = area
        if kind == "room":
            visible |= self.room_tiles.get(idx, set())
        else:
            visible |= self.corridor_tiles.get(idx, set())
            visible |= self.corridor_doors.get(idx, set())
        return visible

    def visible_snapshot(self, robot: Point) -> str:
        area = self.area_at(robot)
        if area is None:
            return ""
        visible = self.visible_tiles(robot)
        rx, ry = robot
        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        if not xs or not ys:
            return ""
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        rows: List[str] = []
        for y in range(min_y, max_y + 1):
            row = []
            for x in range(min_x, max_x + 1):
                pos = (x, y)
                if pos == robot:
                    row.append("R")
                elif pos == self.key_pos:
                    row.append("K")
                elif pos == self.exit_pos:
                    row.append("E")
                elif pos not in visible:
                    row.append(" ")
                else:
                    row.append(self.grid[y][x])
            rows.append("".join(row))
        return "\n".join(rows)


class RobotGame:
    def __init__(self) -> None:
        self.restart()

    def restart(self) -> None:
        self.dungeon = Dungeon()
        self.robot = self.dungeon.start
        self.turn = 0
        self.finished = False
        self.won = False
        self.inventory: List[str] = []
        self.visited: Set[Point] = {self.robot}
        self.visit_counts: Dict[Point, int] = {self.robot: 1}
        self.last_feedback = ""
        self.last_event = ""
        self._last_seen_key_logged = False
        self._last_seen_exit_logged = False
        self._last_has_key_logged = False
        self._last_room_count_logged = 0
        self.agent: Optional[RoomAwareAgent] = None

    @property
    def has_key(self) -> bool:
        return "key" in self.inventory

    def execute(self, action: str) -> str:
        if self.finished:
            return "Game already finished."

        self.turn += 1

        if action in DIRS:
            dx, dy = DIRS[action]
            nx, ny = self.robot[0] + dx, self.robot[1] + dy
            if self.dungeon.is_walkable(nx, ny):
                self.robot = (nx, ny)
                self.visited.add(self.robot)
                self.visit_counts[self.robot] = self.visit_counts.get(self.robot, 0) + 1
                if self.robot == self.dungeon.exit_pos:
                    if self.has_key:
                        self.finished = True
                        self.won = True
                        return "Reached the exit and escaped the dungeon!"
                    return "The exit is locked."
                return f"Moved to {self.robot}."
            return "Bumped into a wall."

        if action == "scan":
            return self.dungeon.visible_snapshot(self.robot)

        if action == "pick_up":
            item = self.dungeon.items.pop(self.robot, None)
            if item:
                self.inventory.append(item)
                return f"Picked up {item}!"
            return "Nothing here to pick up."

        if action == "unlock_exit":
            if self.robot == self.dungeon.exit_pos and self.has_key:
                self.finished = True
                self.won = True
                return "Unlocked the exit and escaped the dungeon!"
            return "Nothing to unlock."

        if action == "wait":
            return "Waited."

        return "Unknown action."


class RoomAwareAgent:
    def __init__(self, model: str = "gemini-3.5-flash"):
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        self.known: Dict[Point, str] = {}
        self.room_memories: Dict[int, RoomMemory] = {}
        self.corridor_memories: Dict[int, CorridorMemory] = {}

        self.current_area: Optional[Area] = None
        self.previous_area: Optional[Area] = None
        self.previous_pos: Optional[Point] = None
        self.last_action: Optional[str] = None
        self.last_robot: Optional[Point] = None
        self.last_invalid_moves: Dict[Point, Set[str]] = defaultdict(set)
        self.tile_stack: List[Point] = []
        self.room_stack: List[int] = []
        self.current_target: Optional[Point] = None
        self.current_target_label: str = ""
        self.corridor_entry_room: Optional[int] = None
        self.current_room_entry_door: Optional[Point] = None
        self.current_room_entry_cluster: Optional[frozenset[Point]] = None
        self.room_entry_lock_room: Optional[int] = None
        self.room_entry_lock_door: Optional[Point] = None
        self.room_entry_lock_until_turn: int = 0
        self.corridor_target: Optional[Point] = None
        self.corridor_target_corridor: Optional[int] = None
        self.corridor_route: List[Point] = []
        self.corridor_route_key: Optional[Tuple[int, Point, Point]] = None
        self.last_reason: str = ""
        self.seen_key: Optional[Point] = None
        self.seen_exit: Optional[Point] = None
        self.area_history: List[Area] = []
        self.area_transition_streaks: Dict[Tuple[Area, Area], int] = defaultdict(int)
        self.last_frontier_turn: int = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def adjacent(self, pos: Point) -> List[Point]:
        x, y = pos
        return [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

    def bfs(self, start: Point, goal: Point, game: RobotGame) -> List[Point]:
        if start == goal:
            return [start]

        q: Deque[Point] = deque([start])
        came_from: Dict[Point, Optional[Point]] = {start: None}

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for nxt in self.adjacent(cur):
                if nxt in came_from:
                    continue
                if nxt != goal and not game.dungeon.is_walkable(*nxt):
                    continue
                if nxt != goal and nxt not in self.known:
                    continue
                came_from[nxt] = cur
                q.append(nxt)

        if goal not in came_from:
            return []

        path = [goal]
        cur = goal
        while came_from[cur] is not None:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    def step_action(self, start: Point, step: Point) -> str:
        dx = step[0] - start[0]
        dy = step[1] - start[1]
        return DIR_FROM_DELTA.get((dx, dy), "wait")

    def observe(self, game: RobotGame) -> None:
        robot = game.robot
        area = game.dungeon.area_at(robot)
        self.previous_area = self.current_area
        self.current_area = area

        visible = game.dungeon.visible_tiles(robot)
        new_tiles = 0
        frontier_gain = 0
        for pos in visible:
            x, y = pos
            tile = game.dungeon.grid[y][x]
            if pos not in self.known:
                new_tiles += 1
                # Treat new rooms and new doors as meaningful frontier progress.
                # Corridors can produce lots of new tiles while still leading to
                # the same dead-end, so we do not let corridor floor tiles reset
                # the stall timer.
                if tile == DOOR:
                    frontier_gain += 1
                elif area is not None and area[0] == "room":
                    frontier_gain += 1
            self.known[pos] = tile
            if pos == game.dungeon.key_pos:
                self.seen_key = pos
            if pos == game.dungeon.exit_pos:
                self.seen_exit = pos

        if frontier_gain > 0:
            self.last_frontier_turn = game.turn

        # Breadcrumbs.
        if self.last_robot is None:
            self.tile_stack = [robot]
        elif self.last_action in DIRS:
            if robot == self.last_robot:
                if game.last_feedback.startswith("Bumped into a wall"):
                    self.last_invalid_moves[robot].add(self.last_action)
            else:
                if self.tile_stack and len(self.tile_stack) >= 2 and robot == self.tile_stack[-2]:
                    self.tile_stack.pop()
                else:
                    self.tile_stack.append(robot)

        self.last_robot = robot
        self._record_area_transition(area)

        # Room/corridor memory.
        if area is not None:
            kind, idx = area
            if kind == "room":
                mem = self.room_memories.get(idx)
                if mem is None:
                    mem = RoomMemory(room_id=idx, first_seen_turn=game.turn)
                    self.room_memories[idx] = mem
                mem.tiles_seen.update(visible)
                mem.doors_seen.update(p for p in visible if game.dungeon.grid[p[1]][p[0]] == DOOR)
                if robot in game.dungeon.room_doors.get(idx, set()):
                    mem.doors_seen.add(robot)
                if game.dungeon.key_pos in visible:
                    mem.key_seen = game.dungeon.key_pos
                if game.dungeon.exit_pos in visible:
                    mem.exit_seen = game.dungeon.exit_pos

                if idx not in self.room_stack:
                    self.room_stack.append(idx)
                else:
                    while self.room_stack and self.room_stack[-1] != idx:
                        self.room_stack.pop()

            else:
                cmem = self.corridor_memories.get(idx)
                if cmem is None:
                    cmem = CorridorMemory(corridor_id=idx)
                    self.corridor_memories[idx] = cmem
                cmem.tiles_seen.update(visible)
                cmem.doors_seen.update(game.dungeon.corridor_doors.get(idx, set()))
                cmem.doors_seen.update(p for p in visible if game.dungeon.grid[p[1]][p[0]] == DOOR)

        self._refresh_exploration_flags(game)

    def _refresh_exploration_flags(self, game: RobotGame) -> None:
        for rid, mem in self.room_memories.items():
            room_doors = game.dungeon.room_doors.get(rid, set())
            if room_doors and room_doors.issubset(mem.doors_used):
                mem.explored = True
            elif not room_doors:
                mem.explored = True

        for cmem in self.corridor_memories.values():
            if cmem.doors_seen and cmem.doors_seen.issubset(cmem.tiles_seen):
                cmem.explored = True

    def _corridor_remaining_doors(self, game: RobotGame, corridor_id: int) -> Set[Point]:
        cmem = self.corridor_memories.get(corridor_id)
        all_doors = set(game.dungeon.corridor_doors.get(corridor_id, set()))
        if cmem is not None:
            all_doors |= set(cmem.doors_seen)
        if cmem is None:
            return all_doors
        return all_doors - cmem.doors_visited

    def mark_door_used_if_needed(self, game: RobotGame) -> None:
        if self.previous_area is None or self.current_area is None:
            return
        if self.previous_pos is None:
            return

        prev_kind, prev_idx = self.previous_area
        cur_kind, cur_idx = self.current_area

        if prev_kind == "room" and cur_kind == "corridor":
            prev_mem = self.room_memories.get(prev_idx)
            if prev_mem is not None and self.previous_pos in game.dungeon.room_doors.get(prev_idx, set()):
                prev_mem.doors_used.add(self.previous_pos)
                prev_mem.door_attempt_counts[self.previous_pos] = prev_mem.door_attempt_counts.get(self.previous_pos, 0) + 1
                cluster = self._door_cluster_for_room(game, prev_idx, self.previous_pos)
                prev_mem.doors_used.update(cluster)
                self.corridor_entry_room = prev_idx
                self.current_room_entry_door = self.previous_pos
                self.current_room_entry_cluster = cluster
                if self.previous_pos not in self.known:
                    self.known[self.previous_pos] = DOOR

                cmem = self.corridor_memories.setdefault(cur_idx, CorridorMemory(corridor_id=cur_idx))
                cmem.doors_seen.add(self.previous_pos)
                cmem.doors_visited.add(self.previous_pos)
                cmem.door_attempt_counts[self.previous_pos] = cmem.door_attempt_counts.get(self.previous_pos, 0) + 1

                self.corridor_target = self._corridor_target_for(game, cur_idx)
                self.corridor_target_corridor = cur_idx
                self.current_target = self.corridor_target
                self.current_target_label = (
                    f"cross corridor to {self.corridor_target}" if self.corridor_target is not None else "cross corridor"
                )
                self.corridor_route = []
                self.corridor_route_key = None
                if self.corridor_target is not None:
                    self.corridor_route = self.corridor_path(game.robot, self.corridor_target, game)
                    if len(self.corridor_route) >= 2:
                        self.corridor_route_key = self._corridor_route_key(cur_idx, game.robot, self.corridor_target)

        if prev_kind == "corridor" and cur_kind == "room":
            cur_mem = self.room_memories.setdefault(
                cur_idx,
                RoomMemory(room_id=cur_idx, first_seen_turn=game.turn),
            )
            # The tile we land on when leaving a corridor is the entry door of
            # the new room. Mark it as used immediately so the room logic does
            # not try to send the robot straight back where it came from.
            if game.robot in game.dungeon.room_doors.get(cur_idx, set()):
                cur_mem.doors_used.add(game.robot)
                cur_mem.door_attempt_counts[game.robot] = cur_mem.door_attempt_counts.get(game.robot, 0) + 1
                cur_mem.doors_seen.add(game.robot)
                self.current_room_entry_door = game.robot
                self.current_room_entry_cluster = self._door_cluster_for_room(game, cur_idx, game.robot)
                self.room_entry_lock_room = cur_idx
                self.room_entry_lock_door = game.robot
                self.room_entry_lock_until_turn = game.turn + 6

            if self.previous_area is not None and self.previous_area[0] == "corridor":
                corridor_mem = self.corridor_memories.setdefault(
                    self.previous_area[1],
                    CorridorMemory(corridor_id=self.previous_area[1]),
                )
                corridor_mem.doors_seen.add(game.robot)
                corridor_mem.doors_visited.add(game.robot)
                corridor_mem.door_attempt_counts[game.robot] = corridor_mem.door_attempt_counts.get(game.robot, 0) + 1

            if self.corridor_entry_room is not None and self.corridor_entry_room != cur_idx:
                self.room_memories.setdefault(
                    self.corridor_entry_room,
                    RoomMemory(room_id=self.corridor_entry_room, first_seen_turn=game.turn),
                ).connected_rooms.add(cur_idx)
                cur_mem.connected_rooms.add(self.corridor_entry_room)
            self.corridor_entry_room = None
            self.corridor_target = None
            self.corridor_target_corridor = None
            self.corridor_route = []
            self.corridor_route_key = None
            self.current_target = None
            self.current_target_label = ""

        current_area = game.dungeon.area_at(game.robot)
        if self.room_entry_lock_room is not None and (
            game.turn > self.room_entry_lock_until_turn
            or current_area is None
            or current_area[0] != "room"
            or current_area[1] != self.room_entry_lock_room
        ):
            self.room_entry_lock_room = None
            self.room_entry_lock_door = None
            self.room_entry_lock_until_turn = 0

    def _corridor_target_for(self, game: RobotGame, corridor_id: int) -> Optional[Point]:
        cmem = self.corridor_memories.get(corridor_id)
        doors: Set[Point] = set(game.dungeon.corridor_doors.get(corridor_id, set()))
        if cmem is not None:
            doors.update(cmem.doors_seen)
        doors = {door for door in doors if game.dungeon.room_at(door) is not None}
        if not doors:
            return None

        pos = game.robot
        entry_door = self.current_room_entry_door
        entry_cluster = self.current_room_entry_cluster
        origin_room = self.corridor_entry_room

        unvisited_doors = [door for door in doors if cmem is None or door not in cmem.doors_visited]
        # When there is still an unvisited doorway in a corridor, do not let
        # an already-tested door win on tie-breaks.
        preferred_doors = sorted(unvisited_doors) if unvisited_doors else sorted(doors)

        candidates: List[Tuple[Tuple[int, int, int, int, int, int, int], Point]] = []
        fallback: List[Tuple[Tuple[int, int, int, int, int, int, int], Point]] = []

        for door in preferred_doors:
            door_room = game.dungeon.room_at(door)
            if door_room is None:
                continue

            door_cluster = self._door_cluster_for_room(game, door_room, door)
            path = self.corridor_path(pos, door, game)
            if len(path) < 2:
                continue

            # The destination of a corridor door is the room on the other side
            # of that specific doorway. In a branching corridor, using some other
            # door in the same corridor as the destination can make the agent
            # chase the wrong branch.
            destination_room = door_room

            destination_mem = self.room_memories.get(destination_room) if destination_room is not None else None
            destination_used_doors = len(destination_mem.doors_used) if destination_mem is not None else 0
            destination_total_doors = len(game.dungeon.room_doors.get(destination_room, set())) if destination_room is not None else 0
            destination_unused_doors = max(0, destination_total_doors - destination_used_doors)
            destination_attempts = 0
            if destination_mem is not None:
                destination_attempts = sum(destination_mem.door_attempt_counts.values())

            # Rooms that still have untouched doors should beat bland new rooms.
            if destination_room is None:
                destination_frontier = 3
            elif destination_unused_doors > 0:
                destination_frontier = 0
            elif destination_mem is None:
                destination_frontier = 1  # unseen room
            elif not destination_mem.explored:
                destination_frontier = 2  # seen but not fully explored
            else:
                destination_frontier = 3  # explored room

            unexplored_doors = destination_unused_doors
            recency = self._room_recency_penalty(destination_room, span=8)

            if destination_room == origin_room and destination_unused_doors <= 0:
                recency += 260
            elif destination_unused_doors > 0:
                recency = max(0, recency - 200)

            unvisited_bias = 0 if cmem is not None and door in cmem.doors_visited else -1
            attempts = 0 if cmem is None else cmem.door_attempt_counts.get(door, 0)

            # In corridors with several nearby doors, prefer the doorway that
            # leads to a room with genuinely new frontier rather than the
            # nearest one or a random tie-break.
            score = (
                destination_frontier,
                unvisited_bias,
                attempts,
                -unexplored_doors,
                recency,
                -len(path),
                -abs(door[0] - pos[0]) - abs(door[1] - pos[1]),
                door[0] * 1000 + door[1],
            )
            item = (score, door)

            if entry_cluster is not None and door_cluster == entry_cluster and len(doors) > 1:
                fallback.append(item)
            else:
                candidates.append(item)

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
        if fallback:
            fallback.sort(key=lambda item: item[0])
            return fallback[0][1]

        # If all doors in the corridor have already been visited, still return
        # the most promising door rather than oscillating between random ones.
        for door in doors:
            if entry_door is None or door != entry_door:
                return door
        return doors[0]

    def _corridor_route_key(self, corridor_id: int, start: Point, target: Point) -> Tuple[int, Point, Point]:
        return (corridor_id, start, target)

    def _ensure_corridor_route(self, game: RobotGame) -> None:
        area = game.dungeon.area_at(game.robot)
        if area is None or area[0] != "corridor" or self.corridor_target is None:
            self.corridor_route = []
            self.corridor_route_key = None
            return

        corridor_id = area[1]
        key = self._corridor_route_key(corridor_id, game.robot, self.corridor_target)
        if self.corridor_route_key == key and self.corridor_route and self.corridor_route[0] == game.robot:
            return

        route = self.corridor_path(game.robot, self.corridor_target, game)
        if len(route) < 2:
            self.corridor_route = []
            self.corridor_route_key = None
            return

        self.corridor_route = route
        self.corridor_route_key = key

    def _transition_key(self, a: Area, b: Area) -> Tuple[Area, Area]:
        return tuple(sorted((a, b), key=lambda item: (item[0], item[1])))  # type: ignore[return-value]

    def _record_area_transition(self, area: Optional[Area]) -> None:
        if area is None:
            return
        if not self.area_history:
            self.area_history.append(area)
            return
        if area == self.area_history[-1]:
            return
        if len(self.area_history) >= 2 and area == self.area_history[-2]:
            key = self._transition_key(self.area_history[-1], area)
            self.area_transition_streaks[key] = min(self.area_transition_streaks[key] + 1, 12)
        self.area_history.append(area)
        if len(self.area_history) > 12:
            self.area_history = self.area_history[-12:]

    def _repeat_transition_penalty(self, current_area: Optional[Area], next_area: Optional[Area]) -> int:
        if current_area is None or next_area is None or current_area == next_area:
            return 0

        penalty = 0
        if len(self.area_history) >= 2 and next_area == self.area_history[-2]:
            key = self._transition_key(current_area, next_area)
            streak = self.area_transition_streaks.get(key, 0)
            # Escalate hard on repeat back-and-forth patterns.
            penalty += 180 + (120 * streak) + (40 * streak * streak)

        recent = list(reversed(self.area_history[-8:]))
        for age, past_area in enumerate(recent, start=1):
            if next_area == past_area:
                penalty += max(0, 240 - (age - 1) * 35)
                break

        return penalty

    def _outer_tile_for_door(self, room: Room, door: Point) -> Point:
        x, y = door
        if x == room.x:
            return (x - 1, y)
        if x == room.x + room.w - 1:
            return (x + 1, y)
        if y == room.y:
            return (x, y - 1)
        return (x, y + 1)

    def _room_recency_penalty(self, room_id: Optional[int], span: int = 8) -> int:
        if room_id is None:
            return 0

        penalty = 0
        recent_rooms = [area[1] for area in reversed(self.area_history[-span * 2 :]) if area[0] == "room"]
        for age, past_room in enumerate(recent_rooms, start=1):
            if past_room == room_id:
                penalty += max(0, 260 - (age - 1) * 40)
                if age <= 2:
                    penalty += 140
                break
        return penalty


    def _room_unused_door_count(self, game: RobotGame, room_id: Optional[int]) -> int:
        if room_id is None:
            return 0
        room_doors = game.dungeon.room_doors.get(room_id, set())
        mem = self.room_memories.get(room_id)
        if mem is None:
            return len(room_doors)
        return len(room_doors - mem.doors_used)

    def _room_has_unseen_tiles(self, game: RobotGame, room_id: int) -> bool:
        return any(tile not in self.known for tile in game.dungeon.room_tiles.get(room_id, set()))

    def _corridor_connected_rooms(self, game: RobotGame, start: Point) -> Set[int]:
        area = game.dungeon.area_at(start)
        if area is None or area[0] != "corridor":
            return set()

        q: Deque[Point] = deque([start])
        seen: Set[Point] = {start}
        rooms: Set[int] = set()

        while q:
            cur = q.popleft()
            cx, cy = cur
            for nxt in ((cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)):
                if nxt in seen:
                    continue
                nxt_area = game.dungeon.area_at(nxt)
                if nxt_area is None:
                    continue
                if nxt_area[0] == "corridor":
                    seen.add(nxt)
                    q.append(nxt)
                elif nxt_area[0] == "room":
                    rooms.add(nxt_area[1])
        return rooms


    def _stalled_turns(self, game: RobotGame) -> int:
        return max(0, game.turn - self.last_frontier_turn)

    def _door_destination_room(self, game: RobotGame, room_id: int, door: Point) -> Optional[int]:
        room = game.dungeon.rooms[room_id]
        outer = self._outer_tile_for_door(room, door)
        outer_area = game.dungeon.area_at(outer)
        if outer_area is None or outer_area[0] != "corridor":
            return None

        corridor_id = outer_area[1]
        corridor_doors = sorted(game.dungeon.corridor_doors.get(corridor_id, set()))
        if not corridor_doors:
            return None

        # Corridors are built as explicit room-to-room links. The destination of
        # a doorway should therefore be the room on the far side of *that same
        # corridor segment*, not any room reachable through the larger corridor
        # graph.
        for other_door in corridor_doors:
            if other_door == door:
                continue
            other_room = game.dungeon.room_at(other_door)
            if other_room is not None:
                return other_room

        # Safety net: if the corridor has been partially observed or malformed,
        # fall back to the best non-current room adjacent to this corridor tile.
        reachable_rooms = self._corridor_connected_rooms(game, outer)
        reachable_rooms.discard(room_id)
        if not reachable_rooms:
            return None

        def room_score(rid: int) -> Tuple[int, int, int, int, int]:
            mem = self.room_memories.get(rid)
            unused = self._room_unused_door_count(game, rid)
            if unused > 0:
                frontier = 0
            elif mem is None:
                frontier = 1
            elif not mem.explored:
                frontier = 2
            else:
                frontier = 3
            recency = self._room_recency_penalty(rid)
            center = game.dungeon.rooms[rid].center
            distance = abs(center[0] - door[0]) + abs(center[1] - door[1])
            return (frontier, -unused, recency, distance, rid)

        return sorted(reachable_rooms, key=room_score)[0]

    def _door_gateway_signature(self, game: RobotGame, room_id: int, door: Point) -> Tuple[str, int]:
        # Keep each physical doorway distinct. The earlier wall-side grouping
        # was too coarse and could cause the agent to bounce between unrelated
        # doors on the same side of a room.
        return ("door", door[0] * 1000 + door[1])

    def _room_door_clusters(self, game: RobotGame, doors: Iterable[Point]) -> List[Set[Point]]:
        clusters_by_sig: Dict[Tuple[str, int], Set[Point]] = defaultdict(set)
        for door in sorted(set(doors)):
            rid = game.dungeon.room_at(door)
            if rid is None:
                continue
            sig = self._door_gateway_signature(game, rid, door)
            clusters_by_sig[sig].add(door)
        return [cluster for cluster in clusters_by_sig.values() if cluster]

    def _door_cluster_for_room(self, game: RobotGame, room_id: int, door: Point) -> frozenset[Point]:
        clusters = self._room_door_clusters(game, game.dungeon.room_doors.get(room_id, set()))
        for cluster in clusters:
            if door in cluster:
                return frozenset(cluster)
        return frozenset({door})

    def _door_attempt_count(self, attempts: Dict[Point, int], door: Point) -> int:
        return attempts.get(door, 0)

    def _door_loop_penalty(self, game: RobotGame, room_id: int, door: Point) -> int:
        room = game.dungeon.rooms[room_id]
        outer = self._outer_tile_for_door(room, door)
        outer_area = game.dungeon.area_at(outer)
        penalty = self._repeat_transition_penalty(game.dungeon.area_at(game.robot), outer_area)
        if outer_area is not None:
            recent = list(reversed(self.area_history[-6:]))
            for age, past_area in enumerate(recent, start=1):
                if outer_area == past_area:
                    penalty += max(0, 120 - (age - 1) * 20)
                    break

        destination_room = self._door_destination_room(game, room_id, door)
        destination_unused = self._room_unused_door_count(game, destination_room)

        # A room with still-open doors should remain attractive even if it is
        # not brand new. Only punish the return heavily when that room is already
        # exhausted.
        if destination_room is not None:
            penalty += self._room_recency_penalty(destination_room)
            if destination_unused <= 0:
                # Strongly discourage door choices that would bounce straight back
                # into a room the robot has just been through.
                penalty += self._room_recency_penalty(destination_room, span=4) * 2
            else:
                penalty = max(0, penalty - 160)
        return penalty

    def _frontier_door_score(
        self,
        game: RobotGame,
        room_id: int,
        door: Point,
        pos: Optional[Point] = None,
        prefer_current_room: bool = False,
    ) -> Tuple[int, int, int, int, int, int]:
        """Score a door by how much unexplored frontier it appears to lead to.

        Lower tuples are better.
        """
        if pos is None:
            pos = game.robot

        mem = self.room_memories.get(room_id)
        destination_room = self._door_destination_room(game, room_id, door)
        destination_mem = self.room_memories.get(destination_room) if destination_room is not None else None
        destination_unused_count = self._room_unused_door_count(game, destination_room)

        # A room with unused doors is more valuable than a completely new room
        # if it is the only place that still offers real branching.
        if destination_room is None:
            destination_frontier = 3
        elif destination_unused_count > 0:
            destination_frontier = 0
        elif destination_mem is None:
            destination_frontier = 1
        elif not destination_mem.explored:
            destination_frontier = 2
        else:
            destination_frontier = 3

        unused_door = 0 if mem is None or door not in mem.doors_used else 1
        unexplored_door = 0 if mem is None or door not in mem.doors_seen else 1
        destination_unused = 0 if destination_unused_count > 0 else 1
        attempt_count = 0 if mem is None else mem.door_attempt_counts.get(door, 0)

        recency = self._room_recency_penalty(destination_room, span=10)
        if destination_room == self.corridor_entry_room:
            recency += 120
        if prefer_current_room and room_id == (self.current_area[1] if self.current_area and self.current_area[0] == 'room' else -1):
            recency = max(0, recency - 80)
        if destination_unused_count > 0:
            recency = max(0, recency - 140)

        distance = abs(door[0] - pos[0]) + abs(door[1] - pos[1])
        return (
            destination_frontier,
            destination_unused,
            attempt_count,
            unused_door,
            unexplored_door,
            recency,
            distance,
        )

    def _best_global_frontier_goal(self, game: RobotGame) -> Optional[Dict]:
        """Pick the best door across all discovered rooms when progress stalls."""
        pos = game.robot
        current_room = game.dungeon.area_at(pos)
        current_room_id = current_room[1] if current_room is not None and current_room[0] == 'room' else None

        scored: List[Tuple[Tuple[int, int, int, int, int, int, int], int, Point]] = []
        for room_id, mem in self.room_memories.items():
            doors = sorted(game.dungeon.room_doors.get(room_id, set()))
            if not doors:
                continue
            for door in doors:
                outer = self._outer_tile_for_door(game.dungeon.rooms[room_id], door)
                if not game.dungeon.is_walkable(*outer):
                    continue

                frontier_score = self._frontier_door_score(
                    game,
                    room_id,
                    door,
                    pos=pos,
                    prefer_current_room=True,
                )
                current_room_bias = 0 if room_id == current_room_id else 1
                jitter = random.randint(0, 4)
                score = (
                    frontier_score[0],
                    frontier_score[1],
                    frontier_score[2],
                    frontier_score[3],
                    frontier_score[4],
                    frontier_score[5],
                    current_room_bias,
                    frontier_score[6] + jitter,
                )
                scored.append((score, room_id, door))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0])
        _, room_id, door = scored[0]
        return {
            'id': f'global_frontier_{room_id}_{door[0]}_{door[1]}',
            'kind': 'tile',
            'target': door,
            'label': f'explore frontier room {room_id} via {door}',
            'loop_penalty': 0,
        }

    def corridor_path(self, start: Point, goal: Point, game: RobotGame) -> List[Point]:
        area = game.dungeon.area_at(start)
        if area is None or area[0] != "corridor":
            return []

        corridor_id = area[1]
        if goal == start:
            return [start]

        q: Deque[Point] = deque([start])
        came_from: Dict[Point, Optional[Point]] = {start: None}

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            cx, cy = cur
            for nxt in ((cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)):
                if nxt in came_from:
                    continue
                if nxt != goal and game.dungeon.corridor_at(nxt) != corridor_id:
                    continue
                if nxt != goal and not game.dungeon.is_walkable(*nxt):
                    continue
                came_from[nxt] = cur
                q.append(nxt)

        if goal not in came_from:
            return []

        path = [goal]
        cur = goal
        while came_from[cur] is not None:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    def candidate_goals(self, game: RobotGame) -> List[Dict]:
        goals: List[Dict] = []
        pos = game.robot

        if pos in game.dungeon.items and game.dungeon.items[pos] == "key":
            goals.append({"id": "pickup_key", "kind": "action", "target": None, "label": "pick up key"})
            return goals

        if game.has_key and pos == game.dungeon.exit_pos:
            goals.append({"id": "unlock_exit", "kind": "action", "target": None, "label": "unlock exit"})
            return goals

        if not game.has_key and self.seen_key is not None:
            goals.append({"id": "go_key", "kind": "tile", "target": self.seen_key, "label": "go to key"})
        if game.has_key and self.seen_exit is not None:
            goals.append({"id": "go_exit", "kind": "tile", "target": self.seen_exit, "label": "go to exit"})

        area = game.dungeon.area_at(pos)
        if area is not None and area[0] == "room":
            rid = area[1]
            mem = self.room_memories.get(rid)
            room_doors = sorted(game.dungeon.room_doors.get(rid, set()))
            entry_cluster = self.current_room_entry_cluster
            clusters = self._room_door_clusters(game, room_doors)
            room_entry_locked = (
                self.room_entry_lock_room == rid
                and game.turn <= self.room_entry_lock_until_turn
                and self._room_has_unseen_tiles(game, rid)
            )

            for idx, cluster in enumerate(clusters):
                if room_entry_locked:
                    # Give freshly entered rooms a few turns of uninterrupted
                    # exploration so they do not immediately ping-pong back out
                    # through the same doorway when a corridor has multiple exits.
                    continue

                cluster_doors = sorted(cluster)
                best_door: Optional[Point] = None
                best_source_door: Optional[Point] = None
                best_score: Optional[Tuple[int, int, int]] = None
                best_loop_penalty = 0
                for door in cluster_doors:
                    destination_room = self._door_destination_room(game, rid, door)
                    loop_penalty = self._door_loop_penalty(game, rid, door) + self._room_recency_penalty(destination_room)

                    # A door the robot has already traversed is still a valid exit.
                    # We only make it less attractive, rather than removing it from
                    # consideration entirely, so rooms never get stranded with no
                    # legal goal.
                    if mem is not None and door in mem.doors_used:
                        loop_penalty += 260

                    if entry_cluster is not None and door in entry_cluster and len(clusters) > 1:
                        loop_penalty += 900

                    outer = self._outer_tile_for_door(game.dungeon.rooms[rid], door)
                    score = (loop_penalty, -abs(door[0] - pos[0]) - abs(door[1] - pos[1]), door[0] * 1000 + door[1])
                    if best_score is None or score < best_score:
                        best_score = score
                        best_door = outer
                        best_source_door = door
                        best_loop_penalty = loop_penalty

                if best_door is not None:
                    goals.append({
                        "id": f"room_{rid}_door_{idx}",
                        "kind": "tile",
                        "target": best_door,
                        "label": f"leave room {rid} via {best_source_door}",
                        "loop_penalty": best_loop_penalty,
                    })

        if area is not None and area[0] == "corridor":
            # Corridors are just bridges between rooms. Once inside one, commit
            # to the best door discovered in that corridor rather than only the
            # corridor endpoints.
            corridor_id = area[1]
            corridor_mem = self.corridor_memories.get(corridor_id)
            corridor_visits = len(corridor_mem.tiles_seen) if corridor_mem is not None else 0
            corridor_seen_doors = len(corridor_mem.doors_seen) if corridor_mem is not None else len(game.dungeon.corridor_doors.get(corridor_id, set()))
            corridor_visited_doors = len(corridor_mem.doors_visited) if corridor_mem is not None else 0
            corridor_remaining = self._corridor_remaining_doors(game, corridor_id)
            if corridor_remaining:
                target = self._corridor_target_for(game, corridor_id)
                if target is not None:
                    goals.append({
                        "id": "corridor_target",
                        "kind": "tile",
                        "target": target,
                        "label": f"cross corridor to {target} ({corridor_visited_doors}/{corridor_seen_doors} doors visited, {len(corridor_remaining)} remaining)",
                        "loop_penalty": 220 + corridor_visits * 12 + len(corridor_remaining) * 60,
                    })
                    return goals

        if not goals:
            goals.append({"id": "scan", "kind": "action", "target": None, "label": "scan"})

        return goals

    def _corridor_candidates(self, game: RobotGame) -> List[Point]:
        pos = game.robot
        blocked = self.last_invalid_moves.get(pos, set())
        candidates: List[Tuple[int, Point]] = []
        for action, (dx, dy) in DIRS.items():
            if action in blocked:
                continue
            nxt = (pos[0] + dx, pos[1] + dy)
            if not game.dungeon.is_walkable(*nxt):
                continue
            if nxt not in self.known:
                # Unknown corridor step is a good exploration move.
                score = 20
            else:
                score = 0
            if nxt not in game.visited:
                score += 10
            if len(self.tile_stack) >= 2 and nxt == self.tile_stack[-2]:
                score -= 20
            if game.dungeon.area_at(nxt) is None:
                score += 5
            candidates.append((score, nxt))
        candidates.sort(key=lambda t: (t[0], random.random()), reverse=True)
        return [p for _, p in candidates]

    def choose_goal(self, game: RobotGame, goals: List[Dict]) -> Dict:
        if not goals:
            return {"id": "scan", "kind": "action", "target": None, "label": "scan"}
        if not self.available:
            return self.heuristic_choice(game, goals)

        prompt = textwrap.dedent(
            f"""
            You are controlling a robot that explores a dungeon room by room.
            The robot can only see the room or corridor segment it currently occupies.
            Choose one high-level goal. Do not choose raw movement directions.

            Current area: {self.current_area}
            Rooms discovered: {sorted(self.room_memories.keys())}
            Key collected: {game.has_key}
            Key seen: {self.seen_key}
            Exit seen: {self.seen_exit}
            Turn: {game.turn}

            Candidate goals:
            {json.dumps(goals, indent=2)}

            Rules:
            - Prefer the key if it is known and not yet collected.
            - If the key is collected, prefer the exit if known.
            - Otherwise explore unused doors in the current room.
            - Only scan if no useful exploration goal exists.
            - Output JSON only: {{"goal_id":"...","reason":"..."}}
            """
        ).strip()

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 128},
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            goal_id = str(parsed.get("goal_id", "")).strip()
            for goal in goals:
                if goal["id"] == goal_id:
                    self.last_reason = str(parsed.get("reason", ""))[:120]
                    return goal
        except Exception:
            pass

        return self.heuristic_choice(game, goals)

    def heuristic_choice(self, game: RobotGame, goals: List[Dict]) -> Dict:
        pos = game.robot

        def score(goal: Dict) -> Tuple[int, int, int]:
            kind = goal.get("kind")
            target = goal.get("target")
            label = goal.get("label", "")
            dist = 0 if target is None else abs(target[0] - pos[0]) + abs(target[1] - pos[1])
            loop_penalty = int(goal.get("loop_penalty", 0))
            if "key" in label:
                base = 1000
            elif "exit" in label:
                base = 900
            elif "door" in label:
                base = 700
            elif "corridor" in label:
                base = 500
            elif "backtrack" in label:
                base = 300
            else:
                base = 100
            base -= loop_penalty
            if kind == "action":
                base += 50
            return (-base, dist, random.randint(0, 3))

        return sorted(goals, key=score)[0]

    def target_is_valid(self, game: RobotGame) -> bool:
        if self.current_target is None:
            return False
        if self.current_target == game.robot:
            return False
        return game.dungeon.is_walkable(*self.current_target)

    def set_new_target(self, game: RobotGame) -> None:
        goals = self.candidate_goals(game)
        chosen = self.choose_goal(game, goals)
        if chosen.get("kind") == "action":
            self.current_target = None
            self.current_target_label = chosen.get("label", "")
        else:
            self.current_target = chosen.get("target")
            self.current_target_label = chosen.get("label", chosen.get("id", ""))

    def local_room_or_corridor_move(self, game: RobotGame) -> str:
        pos = game.robot
        area = game.dungeon.area_at(pos)

        if area is not None and area[0] == "corridor":
            return self._greedy_corridor_step(game)

        blocked = self.last_invalid_moves.get(pos, set())
        candidates: List[Tuple[int, str]] = []

        for action, (dx, dy) in DIRS.items():
            if action in blocked:
                continue
            nxt = (pos[0] + dx, pos[1] + dy)
            if not game.dungeon.is_walkable(*nxt):
                continue

            score = 0
            if nxt not in game.visited:
                score += 8
            if nxt not in self.known:
                score += 6
            visit_count = game.visit_counts.get(nxt, 0)
            if visit_count:
                score -= visit_count * visit_count * 7
            if len(self.tile_stack) >= 2 and nxt == self.tile_stack[-2]:
                score -= 90

            next_area = game.dungeon.area_at(nxt)
            score -= self._repeat_transition_penalty(area, next_area)
            if area is not None and area[0] == "room" and next_area is not None:
                next_kind, next_idx = next_area
                if next_kind == "room":
                    score -= self._room_recency_penalty(next_idx)
                elif next_kind == "corridor":
                    score -= self._room_recency_penalty(self._door_destination_room(game, area[1], nxt))

            if area is not None and area[0] == "room":
                kind, idx = area
                if self.current_room_entry_door is not None and nxt == self.current_room_entry_door:
                    score -= 200
                if nxt in game.dungeon.room_doors.get(idx, set()):
                    mem = self.room_memories.get(idx)
                    if mem is None or nxt not in mem.doors_used:
                        score += 20

            if game.dungeon.area_at(nxt) is None:
                score += 4

            candidates.append((score, action))

        if candidates:
            candidates.sort(key=lambda t: (t[0], random.random()), reverse=True)
            return candidates[0][1]

        return "scan"

    def _greedy_corridor_step(self, game: RobotGame) -> str:
        pos = game.robot
        area = game.dungeon.area_at(pos)
        if area is None or area[0] != "corridor":
            return self.local_room_or_corridor_move(game)

        corridor_id = area[1]
        blocked = self.last_invalid_moves.get(pos, set())

        # Keep one committed target while inside a corridor. This prevents the
        # agent from re-deciding every tick and bouncing between two corridor
        # tiles or nearby rooms.
        corridor_remaining = self._corridor_remaining_doors(game, corridor_id)
        corridor_doors = game.dungeon.corridor_doors.get(corridor_id, set())
        if (
            self.corridor_target is None
            or self.corridor_target_corridor != corridor_id
            or self.corridor_target not in corridor_doors
            or (corridor_remaining and self.corridor_target not in corridor_remaining)
        ):
            self.corridor_target = self._corridor_target_for(game, corridor_id)
            self.corridor_target_corridor = corridor_id if self.corridor_target is not None else None
            self.corridor_route = []
            self.corridor_route_key = None

        if self.corridor_target is None:
            return self.local_room_or_corridor_move(game)

        self._ensure_corridor_route(game)
        if self.corridor_route and self.corridor_route[0] == pos and len(self.corridor_route) > 1:
            return self.step_action(pos, self.corridor_route[1])

        path = self.corridor_path(pos, self.corridor_target, game)
        if len(path) > 1:
            self.corridor_route = path
            self.corridor_route_key = self._corridor_route_key(corridor_id, pos, self.corridor_target)
            return self.step_action(pos, path[1])

        candidates: List[Tuple[int, str]] = []
        for action, (dx, dy) in DIRS.items():
            if action in blocked:
                continue
            nxt = (pos[0] + dx, pos[1] + dy)
            if not game.dungeon.is_walkable(*nxt):
                continue

            nxt_area = game.dungeon.area_at(nxt)
            score = 0
            if nxt == self.corridor_target:
                score += 2000
            if nxt_area is not None and nxt_area[0] == "corridor" and nxt_area[1] == corridor_id:
                score += 400
            if nxt not in game.visited:
                score += 60
            if nxt not in self.known:
                score += 30
            visit_count = game.visit_counts.get(nxt, 0)
            if visit_count:
                score -= visit_count * visit_count * 35
            if self.previous_pos is not None and nxt == self.previous_pos:
                score -= 700
            if len(self.tile_stack) >= 2 and nxt == self.tile_stack[-2]:
                score -= 500
            score -= self._repeat_transition_penalty(area, nxt_area)
            candidates.append((score, action))

        if candidates:
            candidates.sort(key=lambda t: (t[0], random.random()), reverse=True)
            return candidates[0][1]

        return "scan"

    def choose_action(self, game: RobotGame) -> str:
        self.observe(game)
        self.mark_door_used_if_needed(game)

        pos = game.robot
        area = game.dungeon.area_at(pos)

        if pos in game.dungeon.items and game.dungeon.items[pos] == "key":
            self.current_target = pos
            self.current_target_label = "pick up key"
            return "pick_up"

        if game.has_key and pos == game.dungeon.exit_pos:
            self.current_target = pos
            self.current_target_label = "unlock exit"
            return "unlock_exit"

        # Once the key or exit is known, do not let corridor bridging override
        # the actual win path.
        if not game.has_key and self.seen_key is not None:
            self.current_target = self.seen_key
            self.current_target_label = "go to key"
            key_path = self.bfs(pos, self.current_target, game)
            if len(key_path) > 1:
                return self.step_action(pos, key_path[1])

        if game.has_key and self.seen_exit is not None:
            self.current_target = self.seen_exit
            self.current_target_label = "go to exit"
            exit_path = self.bfs(pos, self.current_target, game)
            if len(exit_path) > 1:
                return self.step_action(pos, exit_path[1])

        stalled = self._stalled_turns(game)
        in_corridor = area is not None and area[0] == "corridor"
        corridor_remaining_doors = 0
        if in_corridor:
            corridor_remaining_doors = len(self._corridor_remaining_doors(game, area[1]))

        # Do not let the global recovery routine hijack a corridor that still
        # has untested exits. In branching corridors, the corridor memory and
        # its visited-door counter must get first refusal.
        if stalled >= 45 and not (in_corridor and corridor_remaining_doors > 0):
            frontier_goal = self._best_global_frontier_goal(game)
            if frontier_goal is not None:
                self.current_target = frontier_goal.get("target")
                self.current_target_label = f"Recovery: stalled for {stalled} turns. {frontier_goal.get('label', '')}"
                path = self.bfs(pos, self.current_target, game)
                if len(path) > 1:
                    return self.step_action(pos, path[1])
                if in_corridor:
                    return self._greedy_corridor_step(game)
                return self.local_room_or_corridor_move(game)

        if area is not None and area[0] == "corridor":
            corridor_id = area[1]
            corridor_remaining = self._corridor_remaining_doors(game, corridor_id)
            if (
                self.corridor_target is None
                or self.corridor_target_corridor != corridor_id
                or self.corridor_target not in corridor_remaining
            ):
                self.corridor_target = self._corridor_target_for(game, corridor_id)
                self.corridor_target_corridor = corridor_id if self.corridor_target is not None else None
                self.corridor_route = []
                self.corridor_route_key = None
            if self.corridor_target is not None:
                self.current_target = self.corridor_target
                self.current_target_label = f"cross corridor to {self.corridor_target}"
                self._ensure_corridor_route(game)
                if self.corridor_route and self.corridor_route[0] == pos and len(self.corridor_route) > 1:
                    return self.step_action(pos, self.corridor_route[1])
                path = self.corridor_path(pos, self.corridor_target, game)
                if len(path) > 1:
                    self.corridor_route = path
                    self.corridor_route_key = self._corridor_route_key(corridor_id, pos, self.corridor_target)
                    return self.step_action(pos, path[1])
                return self._greedy_corridor_step(game)

        if self.current_target is not None and not self.target_is_valid(game):
            self.current_target = None
            self.current_target_label = ""

        if area is not None and area[0] == "corridor":
            corridor_id = area[1]
            if self.corridor_target is None or self.corridor_target not in self._corridor_remaining_doors(game, corridor_id):
                self.corridor_target = self._corridor_target_for(game, corridor_id)
                self.corridor_route = []
                self.corridor_route_key = None

        if self.current_target is None:
            self.set_new_target(game)

        if area is not None and area[0] == "room":
            rid = area[1]
            if (
                self.room_entry_lock_room == rid
                and game.turn <= self.room_entry_lock_until_turn
                and self._room_has_unseen_tiles(game, rid)
                and self.current_target is not None
                and self.current_target in game.dungeon.room_doors.get(rid, set())
            ):
                # Let the room breathe for a few turns before leaving again.
                self.current_target = None
                self.current_target_label = ""

        if self.current_target is not None:
            path = self.bfs(pos, self.current_target, game)
            if len(path) > 1:
                return self.step_action(pos, path[1])
            if pos == self.current_target:
                if area is not None and area[0] == "corridor":
                    return self._greedy_corridor_step(game)
                self.current_target = None
                self.current_target_label = ""

        return self.local_room_or_corridor_move(game)

def summarize_goal_label(label: str) -> str:
    normalized = label.strip().lower()
    if not normalized or normalized == 'none':
        return 'idle'
    if 'pick up key' in normalized:
        return 'pick up key'
    if 'unlock exit' in normalized:
        return 'unlock exit'
    if 'go to key' in normalized:
        return 'go to key'
    if 'go to exit' in normalized:
        return 'go to exit'
    if 'cross corridor' in normalized:
        return 'cross corridor'
    if normalized.startswith('leave room') or 'explore frontier' in normalized:
        return 'explore room'
    if normalized.startswith('recovery:'):
        return 'recover'
    return label.strip()


def console_goal_family(goal_label: str) -> str:
    goal = summarize_goal_label(goal_label)
    if goal in {'explore room', 'cross corridor', 'recover'}:
        return 'explore'
    if goal in {'go to key', 'pick up key'}:
        return 'key'
    if goal in {'go to exit', 'unlock exit'}:
        return 'exit'
    return goal


def summarize_event(game: 'RobotGame', agent: 'RoomAwareAgent', action: str, feedback: str, goal_label: str) -> str:
    parts = []
    display_goal = summarize_goal_label(goal_label)
    if display_goal and display_goal != 'idle':
        parts.append(f'goal={display_goal}')
    if action == 'pick_up':
        parts.append('action=pick_up')
    elif action == 'unlock_exit':
        parts.append('action=unlock_exit')
    elif action in DIRS:
        parts.append(f'action={action}')
    elif action:
        parts.append(f'action={action}')

    if 'Picked up key' in feedback:
        parts.append('milestone=key collected')
    if 'Reached the exit and escaped the dungeon' in feedback or 'Unlocked the exit and escaped the dungeon' in feedback:
        parts.append('result=escaped')
    if 'The exit is locked.' in feedback:
        parts.append('status=exit locked')
    if 'Bumped into a wall' in feedback:
        parts.append('status=blocked')

    if agent.seen_key and not getattr(game, '_last_seen_key_logged', False):
        parts.append('milestone=key discovered')
        game._last_seen_key_logged = True
    if agent.seen_exit and not getattr(game, '_last_seen_exit_logged', False):
        parts.append('milestone=exit discovered')
        game._last_seen_exit_logged = True

    if game.has_key and not getattr(game, '_last_has_key_logged', False):
        parts.append('status=key collected')
        game._last_has_key_logged = True

    current_rooms = len(agent.room_memories)
    last_rooms = getattr(game, '_last_room_count_logged', 0)
    if current_rooms > last_rooms:
        parts.append(f'milestone=rooms {current_rooms}/{len(game.dungeon.rooms)}')
        game._last_room_count_logged = current_rooms

    return ' | '.join(parts) if parts else f'action={action}'



class Renderer:
    def __init__(self, game: RobotGame, sprite_path: str = "robot_sprite.png"):
        self.game = game
        self.show_debug = False
        self.sprite_path = sprite_path
        self.tile_size = TILE_SIZE
        self.font = pygame.font.SysFont(None, 22)
        self.small_font = pygame.font.SysFont(None, 18)
        self.big_font = pygame.font.SysFont(None, 34)

        # The asset kit is designed as a matched set, so every sprite is scaled
        # by the same ratio. Square art becomes TILE_SIZE x TILE_SIZE, while
        # the door and exit keep their 1x2 / 2x1 proportions.
        self.wall_sprite = self.load_sprite("Wall Tile.png", (self.tile_size, self.tile_size))
        self.corridor_sprite = self.load_sprite("Corridor Tile.png", (self.tile_size, self.tile_size))
        self.corridor_fill_color = (52, 52, 58)
        self.corridor_shadow_color = (28, 28, 32)
        self.key_sprite = self.load_sprite("Key.png", (self.tile_size, self.tile_size))
        self.robot_sprite = self.load_sprite(sprite_path, (self.tile_size, self.tile_size))
        self.door_horizontal = self.load_sprite("Door.png", (self.tile_size * 2, self.tile_size))
        self.door_vertical = pygame.transform.rotate(self.door_horizontal, 90) if self.door_horizontal else None
        self.exit_sprite = self.load_sprite("Exit.png", (self.tile_size, self.tile_size * 2))

    def load_sprite(self, sprite_path: str, size: Optional[Tuple[int, int]] = None) -> Optional[pygame.Surface]:
        path = Path(sprite_path)
        if not path.exists():
            return None
        try:
            img = pygame.image.load(str(path)).convert_alpha()
            if size is not None:
                img = pygame.transform.smoothscale(img, size)
            return img
        except Exception:
            return None

    def blit_center(self, screen: pygame.Surface, sprite: pygame.Surface, center: Tuple[int, int]) -> None:
        rect = sprite.get_rect(center=center)
        screen.blit(sprite, rect)

    def draw_rect(self, screen: pygame.Surface, color: Tuple[int, int, int], x: int, y: int) -> None:
        pygame.draw.rect(screen, color, (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size))

    def room_color(self, room_id: int) -> Tuple[int, int, int]:
        return ROOM_PALETTE[room_id % len(ROOM_PALETTE)]

    def door_sprite_for(self, room: Room, door: Point) -> Optional[pygame.Surface]:
        x, y = door
        if x in (room.x, room.x + room.w - 1):
            return self.door_horizontal or self.door_vertical
        if y in (room.y, room.y + room.h - 1):
            return self.door_vertical or self.door_horizontal
        return self.door_horizontal or self.door_vertical

    def draw_wall_tile(self, screen: pygame.Surface, x: int, y: int) -> None:
        px = x * self.tile_size
        py = y * self.tile_size
        if self.wall_sprite is not None:
            screen.blit(self.wall_sprite, (px, py))
        else:
            pygame.draw.rect(screen, (40, 40, 45), (px, py, self.tile_size, self.tile_size))

    def draw_corridor_tile(self, screen: pygame.Surface, x: int, y: int) -> None:
        px = x * self.tile_size
        py = y * self.tile_size
        if self.corridor_sprite is not None:
            screen.blit(self.corridor_sprite, (px, py))
        else:
            pygame.draw.rect(screen, self.corridor_fill_color, (px, py, self.tile_size, self.tile_size))
            inner = pygame.Rect(px + 3, py + 3, self.tile_size - 6, self.tile_size - 6)
            pygame.draw.rect(screen, self.corridor_shadow_color, inner)

    def draw_corridor_walls(self, screen: pygame.Surface, dungeon: Dungeon, x: int, y: int) -> None:
        connected = set()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not dungeon.in_bounds(nx, ny):
                continue
            if dungeon.corridor_ids[ny][nx] >= 0:
                connected.add((dx, dy))

        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not dungeon.in_bounds(nx, ny):
                continue
            if dungeon.room_ids[ny][nx] >= 0 or dungeon.corridor_ids[ny][nx] >= 0:
                continue
            if (dx, dy) not in connected:
                self.draw_wall_tile(screen, nx, ny)

    def draw(self, screen: pygame.Surface) -> None:
        g = self.game
        d = g.dungeon
        agent = g.agent

        fog = (12, 12, 16)
        room_border = (40, 40, 45)
        corridor_base = (104, 104, 110)
        ui_bg = (20, 20, 26)
        ui_text = (238, 238, 244)
        accent = (100, 220, 255)
        key_color = (250, 216, 60)
        exit_locked = (170, 60, 60)
        exit_open = (80, 190, 85)

        screen.fill(fog)

        # Discovered world rendering. The agent itself still only sees the
        # current room/corridor segment; this layer is the player's memory of
        # where the agent has already been.
        for y in range(d.height):
            for x in range(d.width):
                pos = (x, y)
                rid = d.room_ids[y][x]
                cid = d.corridor_ids[y][x]
                known = agent is not None and pos in agent.known
                if not known and pos not in g.visited:
                    continue

                tile_type = d.grid[y][x]
                if rid >= 0:
                    room = d.rooms[rid]
                    if tile_type == WALL:
                        if self.wall_sprite is not None:
                            screen.blit(self.wall_sprite, (x * self.tile_size, y * self.tile_size))
                        else:
                            self.draw_rect(screen, room_border, x, y)
                    elif tile_type == DOOR:
                        if self.room_color(rid) is not None:
                            self.draw_rect(screen, self.room_color(rid), x, y)
                        sprite = self.door_sprite_for(room, pos)
                        if sprite is not None:
                            center = (x * self.tile_size + self.tile_size // 2, y * self.tile_size + self.tile_size // 2)
                            self.blit_center(screen, sprite, center)
                        else:
                            self.draw_rect(screen, (160, 110, 60), x, y)
                    else:
                        self.draw_rect(screen, self.room_color(rid), x, y)
                elif cid >= 0:
                    self.draw_corridor_tile(screen, x, y)
                    self.draw_corridor_walls(screen, d, x, y)

                if pos == d.key_pos and known:
                    if self.key_sprite is not None:
                        center = (x * self.tile_size + self.tile_size // 2, y * self.tile_size + self.tile_size // 2)
                        self.blit_center(screen, self.key_sprite, center)
                    else:
                        pygame.draw.circle(screen, key_color, (x * self.tile_size + self.tile_size // 2, y * self.tile_size + self.tile_size // 2), 6)
                elif pos == d.exit_pos and known:
                    if self.exit_sprite is not None:
                        center = (x * self.tile_size + self.tile_size // 2, y * self.tile_size + self.tile_size // 2)
                        self.blit_center(screen, self.exit_sprite, center)
                    else:
                        color = exit_open if g.has_key else exit_locked
                        pygame.draw.rect(screen, color, (x * self.tile_size + 6, y * self.tile_size + 6, self.tile_size - 12, self.tile_size - 12))

        # Robot.
        rx, ry = g.robot
        px, py = rx * self.tile_size, ry * self.tile_size
        if self.robot_sprite is not None:
            screen.blit(self.robot_sprite, (px, py))
        else:
            pygame.draw.circle(screen, (220, 90, 90), (px + self.tile_size // 2, py + self.tile_size // 2), 11)

        # Panel.
        panel_y = d.height * self.tile_size
        panel_h = 136 if self.show_debug else 102
        screen.fill(ui_bg, (0, panel_y, d.width * self.tile_size, panel_h))

        current_area = d.area_at(g.robot)
        area_text = "none"
        if current_area is not None:
            area_text = f"{current_area[0]} {current_area[1]}"

        room_count = len(agent.room_memories) if agent else 0
        corridor_count = len(agent.corridor_memories) if agent else 0
        goal = summarize_goal_label(agent.current_target_label if agent else "")

        progress_text = f"rooms={room_count}/{len(d.rooms)}    key={'yes' if g.has_key else 'no'}    exit={'seen' if agent and agent.seen_exit else 'unknown'}"
        lines = [
            f"Turn {g.turn} / {MAX_TURNS}    Goal: {goal or 'idle'}",
            f"Status: {progress_text}",
        ]
        if self.show_debug:
            lines.append(f"Progress: {progress_text}    Corridors seen: {corridor_count}")
            lines.append("R: regenerate   T: toggle debug text   ESC: quit")

        text_font = self.small_font if self.show_debug else self.font
        text_y = panel_y + 8
        for line in lines:
            surf = text_font.render(line, True, ui_text)
            screen.blit(surf, (10, text_y))
            text_y += 22

        # Side legend.
        if self.show_debug:
            legend_x = d.width * self.tile_size - 230
            legend_y = panel_y + 6
            pygame.draw.rect(screen, (28, 28, 34), (legend_x, legend_y, 220, panel_h - 12), border_radius=8)
            header = self.small_font.render("ROOM LEGEND", True, accent)
            screen.blit(header, (legend_x + 10, legend_y + 8))
            y = legend_y + 28
            for rid in sorted(agent.room_memories.keys() if agent else []):
                color = self.room_color(rid)
                pygame.draw.rect(screen, color, (legend_x + 10, y + 2, 12, 12))
                mem = agent.room_memories[rid]
                label = f"Room {rid} {'(key)' if mem.key_seen else ''}{' (exit)' if mem.exit_seen else ''}{' *' if mem.explored else ''}"
                surf = self.small_font.render(label, True, ui_text)
                screen.blit(surf, (legend_x + 28, y))
                y += 16
                if y > legend_y + panel_h - 20:
                    break

        if g.finished:
            msg = "ROBOT ESCAPED" if g.won else "ROBOT FAILED"
            surf = self.big_font.render(msg, True, ui_text)
            rect = surf.get_rect(center=(d.width * self.tile_size // 2, panel_y - 20))
            screen.blit(surf, rect)
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Robot Dungeon")

    game = RobotGame()
    width_px = game.dungeon.width * TILE_SIZE
    panel_h = 136
    height_px = game.dungeon.height * TILE_SIZE + panel_h
    screen = pygame.display.set_mode((width_px, height_px))

    renderer = Renderer(game, sprite_path="robot_sprite.png")
    clock = pygame.time.Clock()
    agent = RoomAwareAgent()
    game.agent = agent
    move_timer = 0.0
    last_console_goal: Optional[str] = None

    log_path = Path(__file__).with_name("run_log.txt")
    log_file = log_path.open("w", encoding="utf-8")

    running = True
    while running:
        now = pygame.time.get_ticks() / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.restart()
                    agent = RoomAwareAgent()
                    game.agent = agent
                    renderer.game = game
                    move_timer = now
                    game._last_seen_key_logged = False
                    game._last_seen_exit_logged = False
                    game._last_has_key_logged = False
                    game._last_room_count_logged = 0
                    game.last_event = "restarted"
                    last_console_goal = None
                elif event.key == pygame.K_t:
                    renderer.show_debug = not renderer.show_debug

        if not game.finished and game.turn < MAX_TURNS and (now - move_timer) >= ACTION_DELAY:
            previous_goal = summarize_goal_label(agent.current_target_label)
            action = agent.choose_action(game)
            feedback = game.execute(action)
            agent.last_action = action
            agent.previous_pos = agent.last_robot
            game.last_feedback = f"{action}: {feedback}"
            move_timer = now

            current_goal = summarize_goal_label(agent.current_target_label)
            event_line = summarize_event(game, agent, action, feedback, current_goal)
            game.last_event = event_line
            log_file.write(f"Turn {game.turn:03d} | {event_line}\n")
            log_file.flush()
            console_goal = console_goal_family(agent.current_target_label)
            # Console output only reports macro goal changes and the final result.
            if console_goal != last_console_goal:
                print(f"Turn {game.turn:03d} | goal={console_goal}")
                last_console_goal = console_goal
            if game.finished:
                print(f"Turn {game.turn:03d} | result=escaped")
        elif not game.finished and game.turn >= MAX_TURNS:
            game.finished = True
            game.won = False
            game.last_feedback = "Turn limit reached."
            game.last_event = "result=turn limit reached"
            log_file.write(f"Turn {game.turn:03d} | result=turn limit reached\n")
            log_file.flush()

        renderer.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    log_file.close()
    pygame.quit()


if __name__ == "__main__":
    main()
