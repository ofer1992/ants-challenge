"""
In search.py, you will implement generic search algorithms
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


class Node:
    """
    Node class. Encapsulates node in search tree. Each node retains it's state,
    parent, action leading to it, and path cost in search tree.
    """
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def actions_to_node(self):
        node = self
        actions = []
        while node.action:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

    def get_succesors(self, problem):
        return [Node(state, self, action, self.cost +step_cost)
                for state, action, step_cost in problem.get_successors(self.state)]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    frontier = util.Stack()
    explored = set()
    frontier.push(Node(problem.get_start_state()))
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.is_goal_state(node.state):
            return node.actions_to_node()
        explored.add(node.state)
        for child in node.get_succesors(problem):
            if child.state not in explored and child not in frontier.list:
                frontier.push(child)

    return []

def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    frontier = util.Queue()
    explored = set()
    frontier.push(Node(problem.get_start_state()))

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.is_goal_state(node.state):
            return node.actions_to_node()
        explored.add(node.state)
        for child in node.get_succesors(problem):
            if child.state not in explored and child not in frontier.list:
                if problem.is_goal_state(child.state):
                    return child.actions_to_node()
                frontier.push(child)

    return []


class PriorityQueueDecorator:
    """
    Decorator for util.PriorityQueue. adds functionality for membership testing
    and retrieval of priority.
    """
    def __init__(self, queue):
        self.pq = queue
        # dictionary of states and their priority
        self.priority_dict = dict()

    def push(self, item, priority):
        self.priority_dict[item.state] = priority
        self.pq.push(item, priority)

    def pop(self):
        item = self.pq.pop()
        self.priority_dict.pop(item.state)
        return item

    def get_priority(self, item):
        return self.priority_dict[item.state]

    def isEmpty(self):
        return self.pq.isEmpty()

    def __contains__(self, item):
        return item.state in self.priority_dict


def best_first_search(problem, f):
    """
    General best first search according to priority function f.
    :param problem:
    :param f:
    :return:
    """
    frontier = PriorityQueueDecorator(util.PriorityQueue())
    explored = set()
    frontier.push(Node(problem.get_start_state()), 0)

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.is_goal_state(node.state):
            return node.actions_to_node()
        explored.add(node.state)

        for child in node.get_succesors(problem):
            if child.state not in explored and child not in frontier:
                frontier.push(child, f(child))
            elif child in frontier:
                if f(child) < frontier.get_priority(child):
                    frontier.push(child, f(child))

    return []

def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    return best_first_search(problem, lambda n: n.cost)

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    return best_first_search(problem, lambda n: n.cost + heuristic(n.state, problem))


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
