import random
import math
import numpy as np
from Polygon import Polygon
import time


SCALAR = 10
NUM_MOVES = 2000


class State2:
    def __init__(self, vertices, regions, r=None, alpha=[200000, 2]):
        self.size = np.size(vertices, 0)
        self.vertices = vertices
        self.regions = regions
        self.alpha = alpha
        self.polygon = Polygon(self.vertices, self.regions)
        self.num_of_edges = sum(len(i) for i in self.regions)

        if r is None: 
            r = self.polygon.perimeter / (5 * self.num_of_edges)
        self.r = r

    def next_station(self):
        while True:
            try:
                tmp = np.random.uniform(-1, 1, (self.size, 2))
                tmp[0] = [0, 0]
                length = np.sqrt(np.sum(tmp * tmp))
                tmp /= length
                nextmove = tmp * np.random.rand() * self.r
                tmp = self.vertices + nextmove
                tmp = np.mod(tmp + np.pi, 2 * np.pi) - np.pi
                next = State2(vertices=tmp, regions=self.regions, r=self.r * 0.9, alpha=self.alpha)
                return next
            except:
                continue

    def reward(self) -> float:
        return (self.alpha[0] * np.var(self.polygon.areas) + 
                self.polygon.perimeter**self.alpha[1] +
                self.alpha[2] * np.var(self.polygon.edges))


class Node2:
    def __init__(self, state: State2, parent=None):
        self.reward = 1e18
        self.visits = 1
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state: State2):
        child = Node2(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward = min(self.reward, reward)
        self.visits += 1

    def fully_expanded(self) -> bool:
        num_moves = NUM_MOVES
        if len(self.children) == num_moves:
            return True
        
        return False


def delete2(node: Node2):
    for it in node.children:
        delete2(it)
    del node


def uctsearch2(budget, root: Node2) -> Node2:
    best_result = root
    best_value = defaultpolicy(root.state)
    for iter in range(int(budget)):
        front = treepolicy(root)
        reward = defaultpolicy(front.state)
        backup(front, reward)
        if reward < best_value:
            best_value = reward
            best_result = front
        if iter % 10000 == 9999:
            print(f"iter:{iter + 1}: {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
            print(f"\tAreas: {np.round(best_result.state.polygon.areas, 6).tolist()}")
            print(f"\tTotal perimeter: {best_result.state.polygon.perimeter / 2}")
    return best_result


def treepolicy(node: Node2) -> Node2:
    while 1:
        if len(node.children) == 0:
            return expand(node)
        elif random.uniform(0, 1) < 0.5:
            node = bestchild(node, SCALAR)
        else:
            if node.fully_expanded() is False:
                return expand(node)
            else:
                node = bestchild(node, SCALAR)


def expand(node: Node2) -> Node2:
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_station()
    while new_state in tried_children:
        new_state = node.state.next_station()

    node.add_child(new_state)
    return node.children[-1]


def bestchild(node: Node2, scalar) -> Node2:
    bestscore = -1e18
    bestchildren = []
    for c in node.children:
        exploit = c.reward
        explore = math.sqrt(2 * math.log(node.visits) / c.visits)
        score = -exploit + scalar * explore
        if score == bestscore:
            bestchildren.append(c)
        elif score > bestscore:
            bestchildren = [c]
            bestscore = score

    assert len(bestchildren) != 0, "OOPS: no best child found, probably fatal"
    return random.choice(bestchildren)


def defaultpolicy(state: State2) -> float:
    return state.reward()


def backup(node: Node2, reward):
    while node is not None:
        node.visits += 1
        node.reward = min(node.reward, reward)
        node = node.parent
