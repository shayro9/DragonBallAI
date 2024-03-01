import math

import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

PRECISION = 1


class BFSAgent():
    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        agent = env.get_initial_state()
        if env.is_final_state(agent):
            return [], 0, 0
        goals = [g[0] for g in env.get_goal_states()]
        open_nodes = [agent]
        closed_nodes = []
        price_dict = {agent: [0, []]}
        while open_nodes:
            agent = open_nodes.pop()
            closed_nodes.append(agent)
            for action, child in env.succ(agent).items():
                env.reset()
                env.set_state(agent)
                price = child[1]
                if child[0] is None or agent[0] in goals:
                    continue
                new_value = price + price_dict[agent][0]
                new_step = env.step(action)
                new_child = new_step[0]
                if not new_child in closed_nodes and not new_child in open_nodes:
                    actions_untill_now = price_dict[agent][1].copy()
                    actions_untill_now.append(action)
                    price_dict[new_child] = [new_value, actions_untill_now]
                    if env.is_final_state(new_child):
                        return price_dict[new_child][1], new_value, len(closed_nodes)
                    open_nodes.insert(0, new_child)
        return [], 0, 0


class AStarNode:
    def __init__(self, state, parent, action, g, f):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.f = f

    def get_path(self):
        path = []
        n = self
        while n.parent is not None:
            path.insert(0, n.action)
            n = n.parent
        return path

    def __eq__(self, other):
        return self.state == other.state

    def __gt__(self, other):
        if self.f == other.f:
            return self.state[0] > other.state[0]
        return self.f > other.f

    def __hash__(self):
        return hash(self.state)


class WeightedAStarAgent():
    def __init__(self) -> None:
        self.cols = 0
        self.heuristic_targets = []

    def manhattan_dist(self, s, g):
        row_dist = abs(s // self.cols - g // self.cols)
        col_dist = abs(s % self.cols - g % self.cols)
        return row_dist + col_dist

    def msap_heuristic(self, state):
        ops = [self.manhattan_dist(state, g) for g in self.heuristic_targets]
        return min(ops)

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        w = PRECISION * h_weight
        self.cols = env.ncol
        agent = env.get_initial_state()

        goals = [g[0] for g in env.get_goal_states()]
        d1 = [env.d1[0]] if not agent[1] else []
        d2 = [env.d2[0]] if not agent[2] else []
        self.heuristic_targets = goals + d1 + d2
        h = self.msap_heuristic(agent[0])

        node = AStarNode(agent, None, 0, 0, w * h)
        open_nodes = heapdict.heapdict()
        open_nodes[agent] = node
        closed_nodes = {}
        popped_closed_nodes = 0
        while open_nodes:
            agent, curr_node = open_nodes.popitem()

            if env.is_final_state(agent):
                path = curr_node.get_path()
                return path, curr_node.g, len(closed_nodes) + popped_closed_nodes

            closed_nodes[agent] = curr_node
            for action, succ in env.succ(agent).items():
                env.reset()
                env.set_state(agent)
                if succ[0] is None or agent[0] in goals:
                    continue
                new_step = env.step(action)
                child_state = new_step[0]

                cost = succ[1]

                d1 = [env.d1[0]] if not child_state[1] else []
                d2 = [env.d2[0]] if not child_state[2] else []
                self.heuristic_targets = goals + d1 + d2

                new_g = curr_node.g + cost
                x = w * self.msap_heuristic(child_state[0])
                y = 0 if w == PRECISION else (PRECISION - w) * new_g
                new_f = y + x

                if child_state not in open_nodes.keys() and child_state not in closed_nodes.keys():
                    new_node = AStarNode(child_state, curr_node, action, new_g, new_f)
                    open_nodes[child_state] = new_node

                elif child_state in open_nodes.keys():
                    new_node = open_nodes[child_state]
                    if new_f < new_node.f:
                        new_node = AStarNode(child_state, curr_node, action, new_g, new_f)
                        open_nodes[child_state] = new_node
                elif child_state in closed_nodes.keys():
                    new_node = closed_nodes[child_state]
                    if new_f < new_node.f:
                        new_node = AStarNode(child_state, curr_node, action, new_g, new_f)
                        open_nodes[child_state] = new_node
                        closed_nodes.pop(child_state)
                        popped_closed_nodes += 1

        return [], 0, 0


class AStarEpsilonNode:
    def __init__(self, state, parent, action, g, f):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.f = f

    def get_path(self):
        path = []
        n = self
        while n.parent is not None:
            path.insert(0, n.action)
            n = n.parent
        return path

    def __eq__(self, other):
        return self.state == other.state

    def __gt__(self, other):
        return self.g > other.g

    def __hash__(self):
        return hash(self.state)


class AStarEpsilonAgent():
    def __init__(self) -> None:
        self.cols = 0
        self.heuristic_targets = []

    def manhattan_dist(self, s, g):
        row_dist = abs(s // self.cols - g // self.cols)
        col_dist = abs(s % self.cols - g % self.cols)
        return row_dist + col_dist

    def msap_heuristic(self, state):
        ops = [self.manhattan_dist(state, g) for g in self.heuristic_targets]
        return min(ops)

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.cols = env.ncol
        agent = env.get_initial_state()

        g = [a[0] for a in env.get_goal_states()]
        d1 = [env.d1[0]] if not agent[1] else []
        d2 = [env.d2[0]] if not agent[2] else []
        self.heuristic_targets = g + d1 + d2

        open_nodes = heapdict.heapdict()
        focal = heapdict.heapdict()
        h = self.msap_heuristic(agent[0])
        node = AStarEpsilonNode(agent, None, 0, 0, h)
        open_nodes[agent] = node
        closed_nodes = {}
        popped_closed_nodes = 0

        while open_nodes:
            min_f = min([v.f for v in open_nodes.values()])
            for s, n in open_nodes.items():
                if n.f <= (1 + epsilon) * min_f:
                    focal[s] = n
            agent, curr_node = focal.popitem()

            open_nodes.pop(agent)
            if env.is_final_state(agent):
                path = curr_node.get_path()
                return path, curr_node.g, len(closed_nodes) + popped_closed_nodes
            closed_nodes[agent] = curr_node
            for action, succ in env.succ(agent).items():
                env.reset()
                env.set_state(agent)
                if succ[0] is None or agent[0] in g:
                    continue
                new_step = env.step(action)
                child_state = new_step[0]

                d1 = [env.d1[0]] if not child_state[1] else []
                d2 = [env.d2[0]] if not child_state[2] else []
                self.heuristic_targets = g + d1 + d2

                cost = succ[1]

                new_g = curr_node.g + cost
                h = self.msap_heuristic(child_state[0])
                new_f = h + new_g

                if child_state not in open_nodes.keys() and child_state not in closed_nodes.keys():
                    new_node = AStarNode(child_state, curr_node, action, new_g, new_f)
                    open_nodes[child_state] = new_node

                elif child_state in open_nodes.keys():
                    new_node = open_nodes[child_state]
                    if new_f < new_node.f:
                        new_node = AStarNode(child_state, curr_node, action, new_g, new_f)
                        open_nodes[child_state] = new_node
                elif child_state in closed_nodes.keys():
                    new_node = closed_nodes[child_state]
                    if new_f < new_node.f:
                        new_node = AStarNode(child_state, curr_node, action, new_g, new_f)
                        open_nodes[child_state] = new_node
                        closed_nodes.pop(child_state)
                        popped_closed_nodes += 1

        return [], 0, 0
