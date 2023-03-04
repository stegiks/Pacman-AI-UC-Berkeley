# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    """
    I use a set to hold the nodes that I have visited
    """
    visited = set() 
    my_stack = util.Stack()
    my_path = util.Stack()

    my_stack.push(problem.getStartState())
    my_path.push([])

    while not my_stack.isEmpty():
        state = my_stack.pop()
        path = my_path.pop()
        visited.add(state)

        if problem.isGoalState(state):
            return path

        for (succ, action, cost) in problem.getSuccessors(state):
            if succ not in visited:
                path_up = path + [action]
                my_stack.push(succ)
                my_path.push(path_up)
    
    return None


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    """
    Visited here is a list because in question 5 the state consised 
    of a tuple that has a set inside and the set is unhashable.
    """
    visited = [problem.getStartState()]
    my_queue = util.Queue()
    my_queue_path = util.Queue()

    my_queue.push(problem.getStartState())
    my_queue_path.push([])

    while not my_queue.isEmpty():
        state = my_queue.pop()
        path = my_queue_path.pop()

        if problem.isGoalState(state):
            return path

        for (succ, action, cost) in problem.getSuccessors(state):
            if succ not in visited:
                path_up = path + [action]
                visited.append(succ)
                my_queue.push(succ)
                my_queue_path.push(path_up)

    return None           

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    """
    pq.pop() returns only the item back so we have to keep
    the cost in the tuple and have it as priority at the same time
    """
    visited = []
    pq = util.PriorityQueue()
    pq_path = util.PriorityQueue()

    pq.push((problem.getStartState(), 0), 0)
    pq_path.push([], 0)

    while not pq.isEmpty():
        state, total_cost = pq.pop()
        path = pq_path.pop()
        if state not in visited:
            visited.append(state)
            if problem.isGoalState(state):
                return path
        
            for (succ, action, cost) in problem.getSuccessors(state):
            
                new_cost = cost + total_cost
                path_up = path + [action]
                pq.push((succ, new_cost), new_cost)
                pq_path.push(path_up, new_cost)
    
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    """
    In astar algorithm the priority is f = cost_up_to_this_point + heuristic 
    """
    visited = []
    pq = util.PriorityQueue()
    pq_path = util.PriorityQueue()

    pq.push((problem.getStartState(), 0), 0)
    pq_path.push([], 0)

    while not pq.isEmpty():
        state, total_cost = pq.pop()
        path = pq_path.pop()
        if state not in visited:
            visited.append(state)
            if problem.isGoalState(state):
                return path
        
            for (succ, action, cost) in problem.getSuccessors(state):
            
                new_cost = cost + total_cost 
                path_up = path + [action]
                priority = new_cost + heuristic(succ, problem)
                pq.push((succ, new_cost), priority)
                pq_path.push(path_up, priority)
    
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
