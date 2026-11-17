# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def registerInitialState(self, state):
        self._visits = {}

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        if not hasattr(self, '_visits'):
            self._visits = {}
        position = gameState.getPacmanPosition()
        self._visits[position] = self._visits.get(position, 0) + 1
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successor = currentGameState.generatePacmanSuccessor(action)
        value = betterEvaluationFunction(successor)
        value -= 2.0 * getattr(self, '_visits', {}).get(successor.getPacmanPosition(), 0)
        if action == Directions.STOP:
            value -= 3.0
        return value

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return _searchAction(self, gameState, "minimax")

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return _searchAction(self, gameState, "alphabeta")

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return _searchAction(self, gameState, "expectimax")

def _searchAction(agent, gameState, mode):
    """Advance depth only after every agent has completed its turn."""
    def value(state, index, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth >= agent.depth:
            return agent.evaluationFunction(state)
        actions = state.getLegalActions(index)
        if not actions:
            return agent.evaluationFunction(state)
        nextIndex = (index + 1) % state.getNumAgents()
        nextDepth = depth + (nextIndex == 0)
        if index == 0:
            result = float('-inf')
            for action in actions:
                child = state.generateSuccessor(index, action)
                result = max(result, value(child, nextIndex, nextDepth, alpha, beta))
                # Strict inequalities preserve the required expansion on ties.
                if mode == "alphabeta":
                    if result > beta:
                        return result
                    alpha = max(alpha, result)
            return result
        if mode == "expectimax":
            total = 0.0
            for action in actions:
                child = state.generateSuccessor(index, action)
                total += value(child, nextIndex, nextDepth, alpha, beta)
            return total / len(actions)
        result = float('inf')
        for action in actions:
            child = state.generateSuccessor(index, action)
            result = min(result, value(child, nextIndex, nextDepth, alpha, beta))
            if mode == "alphabeta":
                if result < alpha:
                    return result
                beta = min(beta, result)
        return result

    if gameState.isWin() or gameState.isLose() or agent.depth <= 0:
        return Directions.STOP
    actions = gameState.getLegalActions(0)
    bestAction, bestValue = Directions.STOP, float('-inf')
    alpha, beta = float('-inf'), float('inf')
    nextIndex = 1 % gameState.getNumAgents()
    for action in actions:
        child = gameState.generateSuccessor(0, action)
        candidate = value(child, nextIndex, int(nextIndex == 0), alpha, beta)
        if bestAction == Directions.STOP and bestValue == float('-inf') or candidate > bestValue:
            bestValue, bestAction = candidate, action
        if mode == "alphabeta":
            alpha = max(alpha, bestValue)
    return bestAction


from functools import lru_cache
from collections import deque


@lru_cache(maxsize=4096)
def _mazeDistances(walls, start):
    """Cache distances on the immutable wall grid, shared by search leaves."""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if (0 <= nx < walls.width and 0 <= ny < walls.height
                    and not walls[nx][ny] and (nx,ny) not in distances):
                distances[nx,ny] = distances[x,y] + 1
                queue.append((nx,ny))
    return distances


def betterEvaluationFunction(currentGameState: GameState):
    """Prefer food progress, avoid active ghosts, and chase reachable scared ghosts.

    Maze distances to food prevent attraction through walls. The score retains
    the game's rewards; remaining food/capsules and proximity shape navigation.
    """
    if currentGameState.isWin():
        return currentGameState.getScore() + 1000000.0
    if currentGameState.isLose():
        return currentGameState.getScore() - 1000000.0
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    walls = currentGameState.getWalls()
    distances = _mazeDistances(walls, (int(position[0]), int(position[1])))
    score = currentGameState.getScore() - 4.0 * len(food)
    if food:
        nearest = min(distances.get(f, 10000) for f in food)
        score += 10.0 / (nearest + 1.0)
    capsules = currentGameState.getCapsules()
    score -= 12.0 * len(capsules)
    danger = False
    for ghost in currentGameState.getGhostStates():
        distance = manhattanDistance(position, ghost.getPosition())
        if ghost.scaredTimer > distance + 1:
            score += 5.0 / (distance + 1.0)
        elif ghost.scaredTimer <= 1:
            danger = True
            if distance <= 1:
                score -= 20.0
            elif distance <= 3:
                score -= 4.0 / distance
    if danger and capsules:
        nearest = min(distances.get(c, 10000) for c in capsules)
        score += 6.0 / (nearest + 1.0)
    return score


# Abbreviation
better = betterEvaluationFunction
