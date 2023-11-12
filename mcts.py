import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.rewards = 0

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.visits += 1
        self.rewards += reward

    def fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_actions())

    def best_child(self, c1=1.0, c2=1.0):
        choices_weights = [
            (c.rewards / c.visits) + c1 * np.sqrt((2 * np.log(self.visits) / c.visits)) + c2 * np.sqrt((2 * np.log(self.visits) / c.visits))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

def tree_policy(node):
    while not node.state.is_terminal():
        if not node.fully_expanded():
            return expand(node)
        else:
            node = node.best_child()
    return node

def expand(node):
    tried_children = [c.state for c in node.children]
    new_state = node.state.get_possible_actions()[0]
    while new_state in tried_children:
        new_state = node.state.get_possible_actions()[1]
    node.add_child(new_state)
    return node.children[-1]

def default_policy(state):
    while not state.is_terminal():
        state = state.get_random_action()
    return state.get_reward()

def backup(node, reward):
    while node is not None:
        node.update(reward)
        node = node.parent

def mcts(root, itermax):
    for i in range(itermax):
        v = tree_policy(root)
        reward = default_policy(v.state)
        backup(v, reward)
    return root.best_child(c_param=0.)


# review process of algorithm