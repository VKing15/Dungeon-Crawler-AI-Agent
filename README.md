# Robot Dungeon – LLM Agent in a Virtual World

## Overview

Robot Dungeon is a partially observable 2D dungeon exploration environment designed to demonstrate an LLM agent harness.

The agent is placed inside a procedurally generated dungeon containing rooms, corridors, doors, a key, and a locked exit. The environment provides observations to the agent, the agent maintains memory of discovered information, selects goals using an LLM planner, and executes actions within the world to complete tasks.

The objective is to:

1. Explore the dungeon
2. Find the key
3. Remember previously discovered locations
4. Locate the exit
5. Escape the dungeon

---

## Features

- Procedurally generated dungeon
- Multiple rooms connected by corridors
- Partial observability (the robot only sees its current room or corridor)
- Persistent memory of discovered rooms and doors
- Key and exit objective
- Gemini LLM integration for high-level reasoning
- Local fallback heuristics when Gemini is unavailable
- Hand-drawn visual assets created in Krita
- Room-and-corridor dungeon designed to encourage memory-based exploration
- Interactive controls for dungeon regeneration and debugging

---

## Observation Format

The agent receives information about:

- Current room or corridor
- Visible doors
- Visible objects (key or exit)
- Inventory state
- Previously discovered rooms
- Previously discovered corridors

The agent does not have access to the entire map. 
This forces the agent to rely on memory rather than perfect information.

---

## Action Space

The robot can perform:

- move_north
- move_south
- move_east
- move_west
- scan
- pick_up
- unlock_exit
- wait

---

## Agent Architecture

Environment
     |
     ▼
Observation Builder
     │
     ▼
Agent Memory
     │
     ▼
Gemini Planner
(or heuristic fallback)
     │
     ▼
Goal Selection
     │
     ▼
Action Execution
     │
     ▼
Environment Update

The LLM selects high-level goals while deterministic pathfinding handles movement.

---

## Running

Install dependencies:

pip install -r requirements.txt

Set a Gemini API key:

Windows CMD:

set GEMINI_API_KEY=YOUR_KEY

PowerShell:

$env:GEMINI_API_KEY="YOUR_KEY"

Run:

python robot_dungeon.py

---

## Controls

R     Regenerate a new dungeon

T     Toggle debug information

ESC   Close the application

---

## Example Task

Goal:

Find the key and escape the dungeon.

Typical behaviour:

Explore rooms
→ Discover key
→ Pick up key
→ Discover exit
→ Navigate to exit
→ Escape dungeon

---

## Design Decisions

### Partial Observability

The robot only observes its current room or corridor segment.

This forces the agent to maintain memory and reason about previously explored areas.

### Room-Level Memory

The agent tracks:

- Rooms visited
- Doors discovered
- Doors used
- Key location
- Exit location

This reduces looping and enables goal-directed exploration.

### LLM Integration

Gemini is used for high-level decision making while deterministic navigation handles movement execution.
When no Gemini API key is available, the system automatically falls back to a deterministic planner so the project remains fully runnable and reproducible.

---

## Challenges Encountered

During development the agent frequently oscillated between rooms and corridors.

Several memory systems were added, including:

- Door visitation tracking
- Corridor commitment behaviour
- Room exploration memory
- Loop avoidance penalties

These improvements significantly increased exploration performance. DESIGN_CHOICES.txt explores this further.

---

## Included Demonstrations

- Run1.mp4
- Run2.mp4
- run_log.txt

The videos demonstrate the agent exploring different dungeon layouts, collecting the key, locating the exit, and completing the task. The log file records goals, actions, discoveries, and milestones throughout execution.
