<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Informed Search - Structured Technical Notes

## CSE 4617: Artificial Intelligence

Hey dude! I've created comprehensive structured notes from your lecture covering all the technical details you asked for. Let me break down the key concepts with examples and code so you can really understand how informed search works[^1_1].

## 1. Search Problem Components

Every search problem needs these **core elements**[^1_1]:

- **State**: Current configuration of the problem
- **Start State**: Initial configuration
- **Goal Test**: Function to check if current state is goal
- **Successor Function**: Generates possible next states
- **Cost**: Path cost from start to current state
- **Total States**: Complete state space size

Think of it like navigating a city - your state is your current location, start state is where you begin, goal test checks if you've reached your destination, and the successor function gives you all the streets you can take from your current position.

## 2. General Tree Search Algorithm

Here's the **fundamental search template** that all search algorithms follow[^1_1]:

```
TREE-SEARCH(problem, strategy) → returns solution or failure
  initialize tree search using initial state of problem
  loop do:
    if no candidates for expansion:
      return failure
    choose leaf node according to strategy
    if chosen node contains goal state:
      return solution
    else:
      expand node & add resulting nodes to search tree
```

The key difference between search algorithms is the **strategy** for choosing which node to expand next.

## 3. Search Heuristics

A **heuristic function** estimates how close a state is to the goal[^1_1]. Think of it as your intuition about which direction to go:

### Mathematical Representation:

- **h(n)** = estimated cost from node n to goal
- **h*(n)** = actual optimal cost from node n to goal

Heuristics are **problem-specific** - what works for one problem won't necessarily work for another. It's like having different types of GPS for different transportation methods.

## 4. Greedy Search

**Greedy search** always expands the node that seems closest to the goal based on the heuristic[^1_1]:

```python
def greedy_search(problem, heuristic):
    '''Greedy search implementation'''
    frontier = PriorityQueue()
    frontier.push(problem.initial, heuristic(problem.initial))
    explored = set()
    
    while not frontier.empty():
        node = frontier.pop()
        
        if problem.goal_test(node.state):
            return node
        
        explored.add(node.state)
        
        for child in node.expand(problem):
            if child.state not in explored:
                frontier.push(child, heuristic(child.state))
    
    return None
```


### Properties[^1_1]:

- **Complete**: Only if maximum depth is finite
- **Optimal**: No - can get trapped by misleading heuristics
- **Time/Space Complexity**: O(b^m) where b=branching factor, m=max depth
- **Behavior**: Like "badly-guided DFS" - rushes toward possibly wrong goals


## 5. A* Search - The Star Algorithm

**A*** combines the best of both worlds by considering both **backward cost** (how much we've spent) and **forward cost** (estimated remaining cost)[^1_1]:

### Core Formula:

**f(n) = g(n) + h(n)**

- **g(n)**: Actual cost from start to node n
- **h(n)**: Heuristic estimate from n to goal
- **f(n)**: Estimated total cost of path through n


### Comparison with Other Algorithms[^1_1]:

- **UCS**: Orders by path cost g(n) only
- **Greedy**: Orders by goal proximity h(n) only
- **A***: Orders by f(n) = g(n) + h(n) (combines both)


### Critical Implementation Detail[^1_1]:

**Stop when goal state is dequeued from fringe**, not when it's added! This ensures optimality.

## 6. Admissible Heuristics

For A* to guarantee optimal solutions, the heuristic must be **admissible**[^1_1]:

**Definition**: A heuristic h is admissible if **0 ≤ h(n) ≤ h*(n)** for all nodes n

This means the heuristic **never overestimates** the true cost to reach the goal.

### A* Optimality Proof[^1_1]:

**Setup**: Let A = optimal goal, B = suboptimal goal, h = admissible heuristic

**Proof Logic**:

1. Let n be any ancestor of A still on the fringe
2. f(n) = g(n) + h(n) ≤ g(A) (since h is admissible)
3. g(A) = f(A) (since h(A) = 0 at goal)
4. f(A) < f(B) (since g(A) < g(B) and both have h=0)
5. Therefore: f(n) ≤ f(A) < f(B)
6. So A and all its ancestors expand before B
7. **A* finds the optimal solution first!**

## 7. Designing Admissible Heuristics

The **key strategy** is **problem relaxation** - remove constraints to make the problem easier[^1_1].

### 8-Puzzle Example:

**Original Problem**[^1_1]:

- States: Tile configurations (9! = 362,880 possible states)
- Actions: Move empty space in 4 directions
- Cost: Number of moves


### Heuristic 1: Misplaced Tiles

**Relaxation**: Pick up and place tiles anywhere
**Heuristic**: Count tiles not in correct position

```python
def misplaced_tiles_heuristic(state, goal):
    '''Count tiles not in correct position'''
    count = 0
    for i in range(len(state)):
        if state[i] != 0 and state[i] != goal[i]:
            count += 1
    return count
```


### Heuristic 2: Manhattan Distance

**Relaxation**: Move tiles without collision
**Heuristic**: Sum of distances each tile must travel

```python
def manhattan_distance_heuristic(state, goal):
    '''Sum of Manhattan distances for each tile'''
    distance = 0
    size = int(len(state) ** 0.5)  # Assuming square puzzle
    
    for i in range(len(state)):
        if state[i] != 0:  # Skip empty space
            # Find current position
            current_row, current_col = i // size, i % size
            
            # Find goal position  
            goal_pos = goal.index(state[i])
            goal_row, goal_col = goal_pos // size, goal_pos % size
            
            # Add Manhattan distance
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    
    return distance

# Example usage:
state = [1, 2, 3, 4, 0, 5, 6, 7, 8]  # Current state
goal =  [1, 2, 3, 4, 5, 6, 7, 8, 0]  # Goal state
print(f'Misplaced tiles: {misplaced_tiles_heuristic(state, goal)}')  # Output: 2
print(f'Manhattan distance: {manhattan_distance_heuristic(state, goal)}')  # Output: 4
```

Both heuristics are **admissible** because they underestimate the actual moves needed.

## 8. Heuristic Dominance and Combination

### Dominance Relationship[^1_1]:

**h₁ dominates h₂ if**: h₁(n) ≥ h₂(n) for all nodes n

- Better heuristic = closer to true cost
- Manhattan distance dominates misplaced tiles


### Combining Heuristics[^1_1]:

**Maximum of admissible heuristics is admissible**:

```python
def combined_heuristic(state, goal):
    '''Combine multiple admissible heuristics'''
    h1 = misplaced_tiles_heuristic(state, goal)
    h2 = manhattan_distance_heuristic(state, goal)
    return max(h1, h2)  # Take maximum for better estimate
```


## 9. Graph Search vs Tree Search

### The Problem[^1_1]:

**Tree search** can expand the same state multiple times via different paths, leading to **exponential wasted work**.

### The Solution[^1_1]:

**Graph Search = Tree Search + History of Expanded States**

```python
def astar_graph_search(problem, heuristic):
    '''A* with graph search to avoid repeated states'''
    frontier = PriorityQueue()
    frontier.push(Node(problem.initial, None, None, 0), heuristic(problem.initial))
    explored = set()  # Track expanded states
    
    while not frontier.empty():
        node = frontier.pop()
        
        if problem.goal_test(node.state):
            return node
        
        if node.state not in explored:  # Only expand if not seen before
            explored.add(node.state)
            
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                child_cost = node.path_cost + problem.step_cost(node.state, action)
                child_node = Node(child_state, node, action, child_cost)
                
                f_cost = child_cost + heuristic(child_state)
                frontier.push(child_node, f_cost)
    
    return None
```


## 10. Consistency of Heuristics

**Problem with Graph Search**: The first path to a state might not be optimal, leading to suboptimal solutions[^1_1].

### Consistency Definition[^1_1]:

**Heuristic h is consistent if**: h(n) ≤ cost(n,n') + h(n')

- For every node n and successor n'
- Also called "monotonic" heuristic
- **Stronger condition than admissibility**


### Why Consistency Matters[^1_1]:

If h is consistent, then f(n) values are **non-decreasing** along any path:

**Proof**:

1. f(n) = g(n) + h(n)
2. f(n') = g(n') + h(n') = g(n) + cost(n,n') + h(n')
3. Since h(n) ≤ cost(n,n') + h(n'):
4. g(n) + h(n) ≤ g(n) + cost(n,n') + h(n')
5. Therefore: **f(n) ≤ f(n')**

This ensures that when we first reach a state, we've found the optimal path to it.

## 11. Complete A* Implementation

Here's a **production-ready A* implementation** with all the technical details:

```python
import heapq
from collections import namedtuple

Node = namedtuple('Node', ['state', 'parent', 'action', 'path_cost'])

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0  # For tie-breaking
    
    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1
    
    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item
    
    def empty(self):
        return len(self.heap) == 0

def astar_search(problem, heuristic):
    '''Complete A* search implementation'''
    initial_node = Node(problem.initial, None, None, 0)
    
    if problem.goal_test(initial_node.state):
        return initial_node
    
    frontier = PriorityQueue()
    frontier.push(initial_node, heuristic(initial_node.state))
    explored = set()
    
    while not frontier.empty():
        node = frontier.pop()
        
        if problem.goal_test(node.state):
            return node  # Found optimal solution!
        
        explored.add(node.state)
        
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            child_cost = node.path_cost + problem.step_cost(node.state, action)
            child_node = Node(child_state, node, action, child_cost)
            
            if child_state not in explored:
                f_cost = child_cost + heuristic(child_state)
                frontier.push(child_node, f_cost)
    
    return None  # No solution found
```


## 12. Worked Example: A* on Grid

Let's trace through A* on a simple grid[^1_1]:

```
Grid (S=Start, G=Goal, #=Obstacle):
S . . #
. # . .
. . . G
```

**Manhattan Distance Heuristic**:

```python
def manhattan_heuristic(state, goal):
    '''Manhattan distance for grid navigation'''
    x1, y1 = state
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)
```

**Step-by-Step Execution**:

1. **Start**: (0,0), g=0, h=5, f=5
2. **Expand (0,0)**: Add (0,1) and (1,0), both with f=5
3. **Expand (0,1)**: Add (0,2) with f=5
4. **Continue** until goal (3,2) is reached

The beauty of A* is that it **balances exploration and exploitation** - it doesn't just rush toward the goal like Greedy, but also considers the cost of getting there.

## 13. Complexity Analysis

| Algorithm | Complete | Optimal | Time | Space |
| :-- | :-- | :-- | :-- | :-- |
| Greedy | No* | No | O(b^m) | O(b^m) |
| A* (Tree) | Yes* | Yes* | O(b^d) | O(b^d) |
| A* (Graph) | Yes* | Yes** | O(b^d) | O(b^d) |

*If heuristic is admissible and space is finite
**If heuristic is consistent[^1_1]

## 14. Key Implementation Tips

### Priority Queue Considerations:

- Use **min-heap** ordered by f(n) values
- Handle **tie-breaking** consistently with counter
- Store **(f_cost, tie_breaker, node)** tuples


### Heuristic Design Process:

1. **Relax the problem** (remove constraints)
2. **Solve relaxed problem optimally**
3. **Use optimal cost as heuristic**
4. **Verify admissibility** (never overestimate)
5. **Test consistency** if using graph search

### Common Pitfalls:

- **Wrong stopping condition**: Stop when goal is **dequeued**, not enqueued
- **Inconsistent heuristic**: Can cause suboptimal solutions in graph search
- **Overestimation**: Makes heuristic inadmissible


## Key Formulas Summary

**Core A* Formula**: f(n) = g(n) + h(n)[^1_1]

**Admissibility**: h(n) ≤ h*(n) for all n[^1_1]

**Consistency**: h(n) ≤ c(n,n') + h(n') for all n, n'[^1_1]

**Dominance**: h₁ dominates h₂ if h₁(n) ≥ h₂(n) for all n[^1_1]

Dude, this covers all the technical details from your lecture! The key insight is that **informed search uses domain knowledge (heuristics) to guide the search more efficiently than blind search**. A* is particularly elegant because it provides **optimality guarantees** while being **computationally efficient** when you have a good heuristic. Let me know if you want me to elaborate on any specific part!

<div style="text-align: center">⁂</div>

[^1_1]: Lecture-2_-Informed-Search.pdf


---

# Extensive Notes on Heuristics: Berkeley AI Course Style

Hey dude! Let me break down everything about heuristics following the Berkeley CS188 approach. Since your teacher follows their curriculum, I'll make sure you understand the patterns and methodologies they emphasize for designing effective heuristics.

## Core Heuristic Concepts

### What Makes a Heuristic Function

A **heuristic function h(n)** estimates the cost from a current state n to the nearest goal state[^2_1][^2_2]. Think of it as your GPS estimate of how much further you need to drive - it's not exact, but it gives you a good sense of progress toward your destination.

**Key Mathematical Properties**:

- **h(n) ≥ 0** for all states n
- **h(goal) = 0** for any goal state
- **h(n)** should approximate **h*(n)** (the true optimal cost)


### The Two Critical Properties

#### Admissibility

A heuristic h is **admissible** if it never overestimates the true cost:
**0 ≤ h(n) ≤ h*(n)** for all nodes n[^2_1][^2_3]

Think of this as being an "optimistic" estimate - you can underestimate how much work is left, but you can never overestimate it. This ensures A* finds optimal solutions.

#### Consistency (Monotonicity)

A heuristic h is **consistent** if for every node n and successor n':
**h(n) ≤ cost(n,n') + h(n')**[^2_1][^2_3]

This is like the triangle inequality - the direct heuristic estimate shouldn't exceed the cost to move plus the new heuristic estimate. Consistency is **stronger** than admissibility and guarantees that A* graph search remains optimal[^2_1].

## Master Pattern: Problem Relaxation

The **fundamental technique** for creating admissible heuristics is **problem relaxation** - systematically removing constraints from the original problem[^2_4][^2_5][^2_6].

### The Relaxation Process

1. **Identify Constraints**: What rules limit your movements/actions?
2. **Remove Constraints**: Which constraints can you ignore?
3. **Solve Relaxed Problem**: Find optimal solution in easier space
4. **Use as Heuristic**: Relaxed solution cost becomes heuristic estimate

**Why This Works**: Any solution to the original problem is also a solution to the relaxed problem, so the relaxed optimal cost is always ≤ original optimal cost[^2_4].

## Pattern Library: Specific Problem Types

### Spatial Navigation Problems

**Problem Pattern**: Moving through grid/maze from start to goal

**Relaxation Strategy**: Remove movement restrictions

#### Manhattan Distance Heuristic

```python
def manhattan_distance(current, goal):
    """Remove wall constraints - move freely in grid"""
    x1, y1 = current
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)
```

**Relaxation Applied**: Ignore walls, move directly in cardinal directions[^2_1][^2_7].

#### Euclidean Distance Heuristic

```python
def euclidean_distance(current, goal):
    """Remove all movement constraints - fly directly"""
    x1, y1 = current
    x2, y2 = goal
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
```

**Relaxation Applied**: Ignore walls AND grid restrictions - move diagonally through any space.

### Sliding Puzzle Problems (8-puzzle, 15-puzzle)

**Problem Pattern**: Rearrange tiles by sliding into empty space

#### Misplaced Tiles Heuristic

```python
def misplaced_tiles(state, goal):
    """Relax: pick up and place tiles anywhere"""
    count = 0
    for i in range(len(state)):
        if state[i] != 0 and state[i] != goal[i]:
            count += 1
    return count
```

**Relaxation Applied**: Remove sliding constraint - teleport tiles directly to destinations[^2_8][^2_9][^2_10].

#### Manhattan Distance for Puzzles

```python
def puzzle_manhattan(state, goal, size):
    """Relax: move tiles through other tiles"""
    distance = 0
    for i in range(len(state)):
        if state[i] != 0:  # Skip empty space
            current_row, current_col = i // size, i % size
            goal_pos = goal.index(state[i])
            goal_row, goal_col = goal_pos // size, goal_pos % size
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance
```

**Relaxation Applied**: Remove collision constraint - tiles can pass through each other[^2_8][^2_11][^2_12].

### Traveling Salesman Problem (TSP)

**Problem Pattern**: Visit all cities exactly once with minimum cost

#### Minimum Spanning Tree (MST) Heuristic

```python
def mst_heuristic(current_city, unvisited_cities, distances):
    """Relax: don't need to return to start, visit each city at most once"""
    # Compute MST of unvisited cities + current city
    # MST cost is admissible lower bound
    return compute_mst_cost(unvisited_cities + [current_city], distances)
```

**Relaxation Applied**: Remove cycle constraint - don't need to return to start[^2_13][^2_14][^2_15].

#### Nearest Neighbor Lower Bound

```python
def nearest_neighbor_bound(current_city, unvisited_cities, distances):
    """Relax: visit cities in any order"""
    total = 0
    for city in unvisited_cities:
        min_distance = min(distances[city][other] for other in unvisited_cities if other != city)
        total += min_distance
    return total
```

**Relaxation Applied**: Remove ordering constraints[^2_14][^2_16].

## Advanced Heuristic Design Patterns

### Pattern Database Heuristics

**Core Idea**: Precompute optimal costs for subproblems, use as heuristics for full problem[^2_4][^2_17].

**Example - 15-Puzzle Pattern Database**:

```python
def pattern_database_heuristic(state, pattern_db, pattern_positions):
    """Use precomputed costs for tile subset"""
    # Extract pattern from current state
    pattern_state = extract_pattern(state, pattern_positions)
    # Look up precomputed cost
    return pattern_db[pattern_state]
```

**When to Use**: Large state spaces where you can identify meaningful subproblems[^2_17][^2_18].

### Composite Heuristics

**Maximum of Multiple Heuristics**:

```python
def composite_max_heuristic(state, goal):
    """Combine multiple admissible heuristics"""
    h1 = manhattan_distance(state, goal)
    h2 = misplaced_tiles(state, goal)
    h3 = pattern_database_lookup(state)
    return max(h1, h2, h3)
```

**Mathematical Guarantee**: If h₁, h₂, ..., hₖ are admissible, then max(h₁, h₂, ..., hₖ) is admissible[^2_1][^2_19].

**Linear Combination** (Advanced):

```python
def weighted_combination(state, goal, weights):
    """Weighted sum - may not preserve admissibility"""
    h1 = manhattan_distance(state, goal) 
    h2 = misplaced_tiles(state, goal)
    return weights[^2_0] * h1 + weights[^2_1] * h2
```

**Warning**: Linear combinations don't necessarily preserve admissibility unless carefully designed[^2_19].

## Heuristic Quality Assessment

### Dominance Relationship

**Definition**: Heuristic h₁ **dominates** h₂ if h₁(n) ≥ h₂(n) for all nodes n[^2_1].

**Performance Implication**: Dominant heuristics expand fewer nodes, leading to faster search.

```python
def compare_heuristics(states, h1_func, h2_func):
    """Check if h1 dominates h2"""
    for state in states:
        if h1_func(state) < h2_func(state):
            return False  # h1 doesn't dominate h2
    return True  # h1 dominates h2
```


### Effective Branching Factor

**Metric**: Measures heuristic quality by comparing actual nodes expanded to theoretical minimum.

**Formula**: If A* expands N nodes to depth d with branching factor b*:
**N = b* + (b*)² + ... + (b*)^d**

Better heuristics have lower effective branching factors[^2_20].

## Problem-Specific Design Strategies

### For Graph/Network Problems

**Pattern**: Shortest path in weighted graphs

**Heuristics**:

- **Straight-line distance** (ignore intermediate nodes)
- **Landmark distances** (precompute distances to key nodes)
- **Hierarchical decomposition** (solve at multiple granularities)


### For Resource Allocation Problems

**Pattern**: Optimize resource distribution with constraints

**Heuristics**:

- **Linear programming relaxation** (remove integer constraints)
- **Greedy bounds** (assume unlimited resources)
- **Capacity ignoring** (remove capacity limits)


### For Scheduling Problems

**Pattern**: Arrange tasks optimally over time

**Heuristics**:

- **Critical path length** (ignore resource conflicts)
- **Earliest completion time** (ignore dependencies)
- **Resource-ignoring bound** (unlimited parallel execution)


## Implementation Best Practices

### Computational Efficiency Trade-offs

```python
class AdaptiveHeuristic:
    def __init__(self, simple_h, complex_h, threshold):
        self.simple_h = simple_h
        self.complex_h = complex_h
        self.threshold = threshold
    
    def evaluate(self, state):
        """Use simple heuristic most of time, complex when needed"""
        if self.should_use_complex(state):
            return self.complex_h(state)
        return self.simple_h(state)
```

**Strategy**: Balance heuristic accuracy vs. computation time. Sometimes a faster, less accurate heuristic wins overall[^2_8].

### Consistency Verification

```python
def verify_consistency(heuristic, problem, sample_states):
    """Check if heuristic satisfies consistency constraint"""
    for state in sample_states:
        for action in problem.actions(state):
            next_state = problem.result(state, action)
            edge_cost = problem.step_cost(state, action)
            
            if heuristic(state) > edge_cost + heuristic(next_state):
                return False  # Violates consistency
    return True
```


## Advanced Techniques from Berkeley Research

### Dynamic Heuristics

**Idea**: Adapt heuristic during search based on observed patterns

```python
class LearningHeuristic:
    def __init__(self, base_heuristic):
        self.base_h = base_heuristic
        self.correction_factors = {}
    
    def evaluate(self, state):
        base_value = self.base_h(state)
        region = self.classify_region(state)
        correction = self.correction_factors.get(region, 0)
        return base_value + correction
    
    def update_correction(self, state, actual_cost):
        """Learn from search experience"""
        region = self.classify_region(state)
        predicted = self.evaluate(state)
        error = actual_cost - predicted
        self.correction_factors[region] = error * 0.1  # Learning rate
```


### Bidirectional Heuristics

**For bidirectional search** - search from both start and goal:

```python
def bidirectional_heuristic(state, start, goal):
    """Estimate remaining cost considering both directions"""
    forward_h = manhattan_distance(state, goal)
    backward_h = manhattan_distance(start, state)
    total_bound = manhattan_distance(start, goal)
    
    # Use triangle inequality to improve estimate
    return max(forward_h, total_bound - backward_h)
```


## Real-World Applications

### Game AI Pathfinding

**Challenge**: Large game maps with dynamic obstacles

**Solutions**:

- **Hierarchical pathfinding**: Use abstract graph heuristics[^2_21]
- **Jump point search**: Skip intermediate nodes in open areas
- **Goal bounding**: Precompute goal regions for better estimates[^2_21]


### Robot Navigation

**Challenge**: Continuous space with complex obstacles

**Solutions**:

- **Sampling-based heuristics**: Use roadmaps for distance estimates[^2_22]
- **Potential field functions**: Gradient-based guidance
- **Multi-resolution planning**: Coarse-to-fine heuristic refinement[^2_22]


## Common Mistakes and How to Avoid Them

### Mistake 1: Overestimating Heuristics

```python
# WRONG - Can overestimate
def bad_heuristic(state, goal):
    straight_line = euclidean_distance(state, goal)
    return straight_line * 1.5  # Overestimate!

# CORRECT - Conservative estimate  
def good_heuristic(state, goal):
    return euclidean_distance(state, goal)  # Never overestimates
```


### Mistake 2: Ignoring Consistency

```python
# Check consistency for graph search
def ensure_consistency(h, problem):
    # Verify triangle inequality holds
    # Fix by reducing heuristic values if needed
    pass
```


### Mistake 3: Over-complex Heuristics

```python
# Balance accuracy vs. speed
def choose_heuristic(problem_size):
    if problem_size < 1000:
        return complex_but_accurate_heuristic
    else:
        return simple_but_fast_heuristic
```


## Summary: The Berkeley Approach

**Core Philosophy**: Good heuristics come from **systematic problem relaxation** and **principled combination of simple estimates**[^2_2][^2_1].

**Design Process**:

1. **Understand your problem structure**
2. **Identify key constraints**
3. **Systematically relax constraints**
4. **Verify admissibility/consistency**
5. **Combine multiple heuristics** when beneficial
6. **Test and iterate** on real problem instances

**Key Insight**: The best heuristic balances **three factors**:

- **Accuracy** (closer to true cost)
- **Efficiency** (fast to compute)
- **Reliability** (consistently good performance)

The Berkeley CS188 approach emphasizes understanding these patterns so you can design effective heuristics for any new problem domain you encounter. The goal isn't just to memorize specific heuristics, but to develop the intuition for systematic heuristic design!

<div style="text-align: center">⁂</div>

[^2_1]: https://inst.eecs.berkeley.edu/~cs188/sp24/assets/notes/cs188-sp24-note03.pdf

[^2_2]: https://inst.eecs.berkeley.edu/~cs188/sp24/

[^2_3]: https://moodle2.units.it/pluginfile.php/583280/mod_resource/content/1/3.2_informed_search_heuristic.pdf

[^2_4]: https://www.cs.toronto.edu/~hojjat/384f06/Lectures/Lecture04-3132.pdf

[^2_5]: https://cw.fel.cvut.cz/old/_media/courses/a4m33pah/03-relaxation.pdf

[^2_6]: https://www.scribd.com/document/493739439/P104-105

[^2_7]: https://www.cs.toronto.edu/~axgao/cs486686_f21/lecture_notes/Lecture_03_on_Heuristic_Search.pdf

[^2_8]: https://inst.eecs.berkeley.edu/~cs188/fa23/assets/lectures/cs188-fa23-lec03.pdf

[^2_9]: https://github.com/jaads/Solving-sliding-puzzle

[^2_10]: https://blog.goodaudience.com/solving-8-puzzle-using-a-algorithm-7b509c331288

[^2_11]: https://www.cs.princeton.edu/courses/archive/spring21/cos226/assignments/8puzzle/specification.php

[^2_12]: https://algo.monster/liteproblems/773

[^2_13]: https://en.wikipedia.org/wiki/Travelling_salesman_problem

[^2_14]: http://www.isid.ac.in/~dmishra/doc/htsp.pdf

[^2_15]: https://leeds-faculty.colorado.edu/glover/fred pubs/429 - TSP - problem heuristics - leading methods, implementations, latest advances.pdf

[^2_16]: https://pubsonline.informs.org/doi/10.1287/mnsc.10.2.225

[^2_17]: https://ai.dmi.unibas.ch/_files/teaching/hs19/po/misc/edelkamp-mochart2006.pdf

[^2_18]: https://cdn.aaai.org/AAAI/2007/AAAI07-185.pdf

[^2_19]: https://www.artiba.org/blog/heuristic-search-ais-problem-solving-tool

[^2_20]: http://homepage.cs.uiowa.edu/~tinelli/classes/145/Fall05/notes/projects.pdf

[^2_21]: https://ojs.aaai.org/index.php/AIIDE/article/view/18740

[^2_22]: http://arxiv.org/pdf/2410.20279.pdf

[^2_23]: Lecture-2_-Informed-Search.pdf

[^2_24]: https://inst.eecs.berkeley.edu/~cs188/fa24/

[^2_25]: https://stackoverflow.com/questions/66682296/berkeley-ai-class-pacman-food-heuristic-without-mazedistance

[^2_26]: https://github.com/oserr/pacman

[^2_27]: https://www.youtube.com/watch?v=Mlwrx7hbKPs

[^2_28]: https://www.scribd.com/presentation/546788874/Informed-Search

[^2_29]: https://ai.berkeley.edu/project_overview.html

[^2_30]: https://github.com/mohammaduzair9/Pacman-Projects

[^2_31]: https://inst.eecs.berkeley.edu/~cs188/su23/assets/notes/cs188-su23-note05.pdf

[^2_32]: https://github.com/mebusy/notes/blob/master/dev_notes/AI_CS188.md

[^2_33]: https://artint.info/2e/html2e/ArtInt2e.Ch3.S6.SS2.html

[^2_34]: https://www.autoblocks.ai/glossary/consistent-heuristic

[^2_35]: https://courses.cs.washington.edu/courses/cse473/18sp/slides/22-Wrapup-1up.pdf

[^2_36]: http://arxiv.org/pdf/1803.06422.pdf

[^2_37]: https://cw.fel.cvut.cz/b212/_media/courses/be4m36pui/pui2022-relax.pdf

[^2_38]: https://www.toolify.ai/ai-news/maximizing-efficiency-heuristic-functions-for-solving-puzzles-1775757

[^2_39]: https://imada.sdu.dk/u/marco/Teaching/AY2012-2013/DM810/Slides/dm810-lec6.pdf

[^2_40]: https://www.sciencedirect.com/science/article/pii/S1877050915004743/pdf?md5=77fe483c4ef5a28aff69d24ca658042e\&pid=1-s2.0-S1877050915004743-main.pdf

[^2_41]: https://www.cs.princeton.edu/~bwk/btl.mirror/tsp.pdf

[^2_42]: https://pubmed.ncbi.nlm.nih.gov/30133477/

[^2_43]: https://www.simplilearn.com/tutorials/artificial-intelligence-tutorial/heuristic-function-in-ai

[^2_44]: https://www.figma.com/community/file/1427942460558756544/heuristic-evaluation-with-ai

[^2_45]: https://study.com/academy/lesson/heuristic-methods-in-ai-definition-uses-examples.html

[^2_46]: https://dl.acm.org/doi/10.1145/3411763.3451727

[^2_47]: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/DAFA3970B77748BA38699C40BDE2516D/S2633776220000424a.pdf/modeling_a_strategic_human_engineering_design_process_humaninspired_heuristic_guidance_through_learned_visual_design_agents.pdf

[^2_48]: https://study.com/academy/lesson/linear-combination-definition-examples.html

[^2_49]: https://faculty.sites.iastate.edu/jia/files/inline-files/8. heuristic functions.pdf

[^2_50]: https://www.youtube.com/watch?v=zRFZwAUQT8U

[^2_51]: https://www.youtube.com/watch?v=ZQd1Oz0xt9U

[^2_52]: https://en.wikipedia.org/wiki/Admissible_heuristic

[^2_53]: https://en.wikipedia.org/wiki/Consistent_heuristic

[^2_54]: https://www.sciencedirect.com/topics/computer-science/relaxed-problem

[^2_55]: https://cs.uky.edu/~sgware/courses/cs660/slides/isearch.pdf

[^2_56]: https://stackoverflow.com/questions/62741107/a-star-search-sliding-tile-puzzle-which-heuristic-is-this

[^2_57]: https://core.ac.uk/download/pdf/53190059.pdf

[^2_58]: https://trace.tennessee.edu/cgi/viewcontent.cgi?article=8031\&context=utk_gradthes

[^2_59]: https://www.youtube.com/watch?v=u1b9QQe0FsA

[^2_60]: https://people.cs.nott.ac.uk/pszrq/files/ANOR11gchh.pdf

[^2_61]: https://stackoverflow.com/questions/36074377/composite-heuristic-for-the-8-puzzle

