import random
import math
import numpy as np
from Voronoi import Voronoi, spherical_to_cartesian
from Structure import Structure
import time

SCALAR = 10
SIZE = 10
NUM_MOVES = 20000


class State1:
    def __init__(self, size=SIZE, points=None, r=None, alpha=[400, 2]):
        self.size = size
        self.alpha = alpha

        if points is None:
            points = np.random.uniform(-np.pi, np.pi, self.size * 2)
        points[0] = points[1] = 0

        self.points = points
        if r is None: 
            r = 2 * np.pi * np.sqrt(size * 2)
        self.r = r

        cartesian_points = np.array([spherical_to_cartesian(self.points[i * 2], self.points[i * 2 + 1])
                                     for i in range(self.size)])
        self.voronoi = Voronoi(cartesian_points)
        self.total_variance = 0
        
    def next_station(self):
        tmp = np.random.uniform(-1, 1, self.size * 2)
        tmp[0] = tmp[1] = 0
        length = np.sqrt(np.sum(tmp * tmp))
        tmp /= length
        nextmove = tmp * np.random.rand() * self.r
        tmp = self.points + nextmove
        tmp = np.mod(tmp + np.pi, 2 * np.pi) - np.pi
        next = State1(size=self.size, points=tmp, r=self.r * 0.8, alpha=self.alpha)
        return next

    def reward(self):
        return (self.alpha[0] * np.var(self.voronoi.areas) + 
                self.voronoi.perimeter**self.alpha[1] + 
                self.alpha[2] * np.var(self.voronoi.edges))


class Node1:
    def __init__(self, state=State1(), parent=None):
        self.reward = 1e18
        self.visits = 1
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state: State1):
        child = Node1(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward = min(self.reward, reward)
        self.visits += 1

    def fully_expanded(self):
        num_moves = NUM_MOVES
        if len(self.children) == num_moves:
            return True
        
        return False


def delete1(node: Node1):
    for it in node.children:
        delete1(it)
    del node


def uctsearch1(budget, root: Node1) -> tuple[Voronoi, float]:
    best_value = defaultpolicy(root.state)
    best_result = root.state.voronoi
    for iter in range(int(budget)):
        front = treepolicy(root)
        reward = defaultpolicy(front.state)
        backup(front, reward)
        if reward < best_value:
            best_value = reward
            best_result = front.state.voronoi
        if iter % 10000 == 9999:
            print(f"iter:{iter + 1}: {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
            print(f"\tstructure:{best_result.structure}")
            print(f"\tvalue:{best_result.perimeter / 2}")
    return best_result, best_value


def treepolicy(node: Node1) -> Node1:
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


def expand(node: Node1) -> Node1:
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_station()
    while new_state in tried_children:
        new_state = node.state.next_station()

    node.add_child(new_state)
    return node.children[-1]


def bestchild(node: Node1, scalar) -> Node1:
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


def defaultpolicy(state: State1) -> float:
    return state.reward()


def backup(node: Node1, reward):
    while node is not None:
        node.visits += 1
        node.reward = min(node.reward, reward)
        node = node.parent
