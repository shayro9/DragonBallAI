import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class BFSAgent():
    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        agent = env.get_initial_state()
        if env.is_final_state(agent):
            return [], 0, 0
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
                terminated = child[2]
                if price == float('inf'):
                    continue
                new_step = env.step(action)
                new_child = new_step[0]
                new_value = price + price_dict[agent][0]
                if not new_child in closed_nodes and not new_child in open_nodes:
                    if not terminated or env.is_final_state(new_child):
                        actions_untill_now = price_dict[agent][1].copy()
                        actions_untill_now.append(action)
                        price_dict[new_child] = [new_value, actions_untill_now]
                        if env.is_final_state(new_child):
                            return price_dict[new_child][1], new_value, len(closed_nodes)
                        open_nodes.insert(0, new_child)

class AStarNode:
    def __init__(self, state, parent, g, f):
        self.state = state
        self.parent = parent
        self.g = g
        self.f = f

    def __eq__(self, other):
        return self.state == other.state

class WeightedAStarAgent():
    def __init__(self) -> None:
        self.cols = 0
        self.heuristic_targets = []


    def manhatan_dist(self, s, g):
        row_dist = abs(s // self.cols - g // self.cols)
        col_dist = abs(s % self.cols - g % self.cols)
        return row_dist + col_dist

    def msap_heuristic(self, state):
        ops = [self.manhatan_dist(state[0], g) for g in self._heuristic_targets]
        return min(ops)

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.cols = env.ncol
        self._heuristic_targets = [a[0] for a in env.get_goal_states()] + [env.d1[0], env.d2[0]]
        start = env.get_initial_state()
        node = self.AStarNode(start, None, 0, h_weight * self.msap_heuristic(start[0]))

        open_nodes = heapdict.heapdict()
        h = self.msap_heuristic(start[0])
        open_nodes[node] = h_weight * h + (1-h_weight) * node.g

        closed_nodes = heapdict.heapdict()
        while open_nodes:
            curr, f_value = open_nodes.popitem()
            closed_nodes[curr] = f_value
            if env.is_final_state(curr):
                return #TODO
            for action, child in env.succ(curr):
                new_g = curr.g + child[1]
                new_f = new_g + self.msap_heuristic(curr[0])
                child_node = node()



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
