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
            return [], 0,0 #TODO:return
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
                if price == float('inf'):
                    continue
                new_step = env.step(action)
                new_child = new_step[0]
                new_value = price + price_dict[agent][0]
                if not new_child in closed_nodes and not new_child in open_nodes:
                    actions_untill_now = price_dict[agent][1].copy()
                    actions_untill_now.append(action)
                    price_dict[new_child] = [new_value, actions_untill_now]
                    print(f"for the node: {new_child} we have the price: {new_value}\n")
                    if env.is_final_state(new_child):
                        return price_dict[new_child][1], new_value, len(closed_nodes)    #TODO:return
                    open_nodes.insert(0, new_child)


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError