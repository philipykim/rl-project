import optparse
import qlearningAgents
import util
import pdb, traceback, sys
import random
from cStringIO import StringIO
import datetime

class Grid:
  """
  A 2-dimensional array of immutables backed by a list of lists.  Data is accessed
  via grid[x][y] where (x,y) are cartesian coordinates with x horizontal,
  y vertical and the origin (0,0) in the bottom left corner.  
  
  The __str__ method constructs an output that is oriented appropriately.
  """
  def __init__(self, width, height, initialValue=' '):
    self.width = width
    self.height = height
    self.data = [[initialValue for y in range(height)] for x in range(width)]
    self.terminalState = 'TERMINAL_STATE'
    
  def __getitem__(self, i):
    return self.data[i]
  
  def __setitem__(self, key, item):
    self.data[key] = item
    
  def __eq__(self, other):
    if other == None: return False
    return self.data == other.data
    
  def __hash__(self):
    return hash(self.data)
  
  def copy(self):
    g = Grid(self.width, self.height)
    g.data = [x[:] for x in self.data]
    return g
  
  def deepCopy(self):
    return self.copy()
  
  def shallowCopy(self):
    g = Grid(self.width, self.height)
    g.data = self.data
    return g
    
  def _getLegacyText(self):
    t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
    t.reverse()
    return t
    
  def __str__(self):
    return str(self._getLegacyText())

def getBridgeGrid():
  # The most basic bridge grid
  grid = [[   'AS',  '#',  '#',  '#',   'BS'],
          [    ' ',  ' ',  ' ',  ' ',    ' '],
          [ 'BE10',  '#',  '#',  '#', 'AE10']]
  return Gridworld(grid)

def getBigBridgeGrid():
  # The point of this one is to test partial observability
  grid  = [[  'AS',   '#',   '#',   '#',   '#',   '#',   'BS'],
           [   ' ',   ' ',   ' ',   ' ',   ' ',   ' ',    ' '],
           ['BE10',   '#',   '#',   '#',   '#',   '#', 'AE10']]
  return Gridworld(grid)

def getBigUglyBridgeGrid():
  # Intended to force communication
  grid  = [[  'AS',   '#',   '#',   '#',   '#',   '#',   'BS'],
           [   ' ', 'Z-1',   ' ',   ' ',   ' ', 'Z-1',    ' '],
           ['BE10',   '#',   '#',   '#',   '#',   '#', 'AE10']]
  def initializer():
    squareToZap = random.choice((1, 1), (5, 1), None)
    if squareToZap:
      x, y = squareToZap
      grid[x][y] = '#'
  def mutator(state):
    if (3, 1) in state.values():
      grid[1][1] = 'Z-10'
      grid[5][1] = 'Z-10'
  return Gridworld(grid, initializer, mutator)

class Gridworld():
  """
    Gridworld encodes the static definition of how the world behaves.
  """
  def __init__(self, gridString, initializer=None, mutator=None, allowComm=False):

  	# Build grids
    width, height = len(gridString[0]), len(gridString)
    grid = Grid(width, height)
    for ybar, line in enumerate(gridString):
      y = height - ybar - 1
      for x, el in enumerate(line):
        grid[x][y] = el
    self.grid = grid

    self.initializer = initializer
    self.mutator = mutator
    self.allowComm = allowComm

    # parameters
    self.livingReward = 0.0
    self.noise = 0.2
        
  def setLivingReward(self, reward):
    """
    The (negative) reward for exiting "normal" states.
    
    Note that in the R+N text, this reward is on entering
    a state and therefore is not clearly part of the state's
    future rewards.
    """
    self.livingReward = reward
        
  def setNoise(self, noise):
    """
    The probability of moving in an unintended direction.
    """
    self.noise = noise

  def getStates(self):
    """
    Return list of all states.
    """
    positions = [self.grid.terminalState]
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if self.grid[x][y] != '#':
          pos = (x,y)
          positions.append(pos)

    # TODO: un-hardwire A and B
    states = []
    for posA in positions:
      protostate = {}
      protostate['A'] = posA
      for posB in positions:
        if posB == posA:
          continue
        state = dict(protostate)
        state['B'] = posB
        states.append(state)

    return states

  def getPossibleActions(self, state, agent):
    """
    Returns list of valid actions for the agent with 
    label 'agent' when the world is in the specified state.
    
    Note that you can request moves into walls and
    that "exit" states transition to the terminal
    state under the special action "done".
    """
    if state[agent] == self.grid.terminalState:
      return ()

    x, y = state[agent]
    cell = self.grid[x][y]

    if cell[0] == agent and cell[1] == 'E':
      return ('exit',)
    
    if self.allowComm:
      return ('north','west','south','east','comm')

    return ('north','west','south','east')

  def getReward(self, state, agent, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    if state[agent] == self.grid.terminalState:
      return 0.0
    x, y = state[agent]
    cell = self.grid[x][y]
    if cell[0] == agent or cell[0] == 'Z':
      if cell[1] == 'E' or cell[1] == 'S':
        if len(cell) > 2:
          return int(cell[2:])
        return self.livingReward
      return int(cell[1:])
    return self.livingReward

  def getStartState(self):
    state = {}
    for x in range(self.grid.width):
      for y in range(self.grid.height):
      	cell = self.grid[x][y]
      	if len(cell) > 1 and cell[1] == 'S':
          state[cell[0]] = (x, y)
    return state

  def isTerminal(self, state, agent):
    """
    Only the TERMINAL_STATE state is *actually* a terminal state.
    The other "exit" states are technically non-terminals with
    a single action "exit" which leads to the true terminal state.
    This convention is to make the grids line up with the examples
    in the R+N textbook.
    """
    return state[agent] == self.grid.terminalState

  def getTransitionStatesAndProbs(self, state, agent, action):
    """
    Returns list of (nextState, prob) pairs
    representing the states reachable
    from 'state' by taking 'action' along
    with their transition probabilities.          
    """

    if action not in self.getPossibleActions(state, agent):
      raise "Illegal action!"
    
    # If the agent is at the terminal state, nothing happens  
    if self.isTerminal(state, agent):
      return [(dict(state), 1.0)]
    
    x, y = state[agent]
    cell = self.grid[x][y]

    # If the agent is at an exit, guaranteed to enter terminal state
    if cell[0] == agent and cell[1] == 'E':
      nextState = dict(state)
      nextState[agent] = self.grid.terminalState
      return [(nextState, 1.0)]

    # TODO: If agent is communicating...

    # If agent is trying to move north, south, east, west, it is subject to noise
    successors = []                

    northState = (self.__isAllowed(state,y+1,x) and (x,y+1)) or state[agent]
    westState = (self.__isAllowed(state,y,x-1) and (x-1,y)) or state[agent]
    southState = (self.__isAllowed(state,y-1,x) and (x,y-1)) or state[agent]
    eastState = (self.__isAllowed(state,y,x+1) and (x+1,y)) or state[agent]
                        
    if action == 'north' or action == 'south':
      if action == 'north': 
        successors.append((northState,1-self.noise))
      else:
        successors.append((southState,1-self.noise))
                                
      massLeft = self.noise
      successors.append((westState,massLeft/2.0))    
      successors.append((eastState,massLeft/2.0))
                                
    if action == 'west' or action == 'east':
      if action == 'west':
        successors.append((westState,1-self.noise))
      else:
        successors.append((eastState,1-self.noise))
                
      massLeft = self.noise
      successors.append((northState,massLeft/2.0))
      successors.append((southState,massLeft/2.0)) 
      
    successors = self.__aggregate(successors)

    nextStates = []
    for pos, prob in successors:
      nextState = dict(state)
      nextState[agent] = pos
      nextStates.append((nextState, prob))

    return nextStates
  
  def __aggregate(self, posAndProbs):
    counter = util.Counter()
    for pos, prob in posAndProbs:
      counter[pos] += prob
    newPosAndProbs = []
    for pos, prob in counter.items():
      newPosAndProbs.append((pos, prob))
    return newPosAndProbs
        
  def __isAllowed(self, state, y, x):
    return not (self.isWall(state, y, x) or self.isAgent(state, y, x))

  def isWall(self, state, y, x):
    if y < 0 or y >= self.grid.height: return True
    if x < 0 or x >= self.grid.width: return True
    return self.grid[x][y] == '#'

  def isAgent(self, state, y, x):
    return (x, y) in state.values()


class GridworldEnvironment():
    
  def __init__(self, gridWorld):
    self.gridWorld = gridWorld
    self.reset()
    self.agentLabels = sorted(self.state.keys())

  def getCurrentState(self):
    return self.state

  def getPossibleActions(self, state, agent):        
    return self.gridWorld.getPossibleActions(state, agent)

  def doAction(self, agent, action):
    successors = self.gridWorld.getTransitionStatesAndProbs(self.state, agent, action) 
    sum = 0.0
    rand = random.random()
    state = self.getCurrentState()
    for nextState, prob in successors:
      sum += prob
      if sum > 1.0:
        raise 'Total transition probability more than one; sample failure.' 
      if rand < sum:
        reward = self.gridWorld.getReward(state, agent, action, nextState)
        self.state = nextState
        return (nextState, reward)
    raise 'Total transition probability less than one; sample failure.'    

  def reset(self):
    self.state = self.gridWorld.getStartState()


def runEpisode(agents, environment, discount, display, message, pause, episode):

  returns = util.Counter()
  totalDiscount = 1.0
  environment.reset()
  for agent in agents: 
  	if 'startEpisode' in dir(agent): 
  	  agent.startEpisode()

  message("BEGINNING EPISODE: "+str(episode)+"\n")

  transitions = [None for a in agents] # tuple of state, action, reward
  timestep = 0
  
  while True:

    completedAgents = 0

    for index, agent in enumerate(agents):
      agentLabel = environment.agentLabels[index]

      state = environment.getCurrentState()

      # UPDATE LEARNER
      if transitions[index] != None and 'observeTransition' in dir(agent):
        previousState, previousAction, previousReward = transitions[index]
        agent.observeTransition(previousState, previousAction, state, previousReward)

      if len(environment.getPossibleActions(state, agentLabel)) == 0:
        transitions[index] = None
        completedAgents += 1
        continue

      # DISPLAY CURRENT STATE
      display(state, agent, agentLabel)
      pause()

      # GET ACTION (USUALLY FROM AGENT)
      action = agent.getAction(state)
      if action == None:
        raise 'Error: Agent returned None action'

      # EXECUTE ACTION
      nextState, reward = environment.doAction(agentLabel, action)
      message("Started in state: "+str(state)+
              "\nTook action: "+str(action)+
              "\nEnded in state: "+str(nextState)+
              "\nGot reward: "+str(reward)+"\n")
      transitions[index] = (state, action, reward)
    
      returns[agentLabel] += reward * totalDiscount

    totalDiscount *= discount
    timestep += 1

    if completedAgents == len(agents):
      message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+" AFTER "+str(timestep)+" TIMESTEPS\n")
      return returns

def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="BridgeGrid",
                         help='Grid to use (case sensitive; options are BridgeGrid, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\' and \'q\', default %default)')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-o', '--partial',action='store_true',
                         dest='partial',default=False,
                         help='Whether the world is partially-observable')
    optParser.add_option('-q', '--quiet',action='store',
                         type='int',dest='quiet',default=0,
                         help='Number of episodes to skip displaying')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store',
                         type='int',dest='manual',default=0,
                         help='Number of manually controlled agents')
    optParser.add_option('-v', '--valueSteps',action='store_true' ,default=False,
                         help='Display each step of value iteration')

    opts, args = optParser.parse_args()
    
    # MANAGE CONFLICTS
    if opts.textDisplay:
      opts.pause = False
      
    if opts.manual > 0:
      opts.pause = True
      
    return opts

class RandomAgent:
  def __init__(self, possibleActions):
  	self.possibleActions = possibleActions
  def getAction(self, state):
    return random.choice(self.possibleActions(state))
  def getValue(self, state):
    return 0.0
  def getQValue(self, state, action):
    return 0.0
  def getPolicy(self, state):
    "NOTE: 'random' is a special policy value; don't use it in your code."
    return 'random'
  def update(self, state, action, nextState, reward):
    pass      

class UserAgent:
  def __init__(self, possibleActions):
  	self.possibleActions = possibleActions
  def getAction(self, state):
    import graphicsUtils
    action = None
    while True:
      keys = graphicsUtils.wait_for_keys()
      if 'Up' in keys: action = 'north'
      if 'Down' in keys: action = 'south'
      if 'Left' in keys: action = 'west'
      if 'Right' in keys: action = 'east'
      if 'q' in keys: sys.exit(0)
      if action == None: 
      	continue
      break
    actions = self.possibleActions(state)
    if action not in actions:
      action = actions[0]
    return action
  def getValue(self, state):
    return 0.0
  def getQValue(self, state, action):
    return 0.0
  def getPolicy(self, state):
    return 'user'
  def update(self, state, action, nextState, reward):
    pass

def printString(x): print x

def flatteningObserver(state):
  orderedState = [state[key] for key in sorted(state.keys())]
  observation = [e for l in orderedState for e in l]
  return tuple(observation)

def radiusOneObserverFactory(world, agent):

  def radiusOneObserver(state):
    if world.isTerminal(state, agent):
      return ""
    observation = StringIO()
    ax, ay = state[agent]
    for x in range(ax-1, ax+2):
      for y in range(ay-1, ay+2):
        if world.isWall(state, y, x):
          observation.write("#")
        elif world.isAgent(state, y, x):
          observation.write("X")
        else:
          observation.write(" ")
    return observation.getvalue()

  return radiusOneObserver

if __name__ == '__main__':
  try:
    opts = parseOptions()
  
    import multiAgentWorld
    worldFunction = getattr(multiAgentWorld, "get" + opts.grid)
    world = worldFunction()
    world.setLivingReward(opts.livingReward)
    world.setNoise(opts.noise)
    env = GridworldEnvironment(world)

    import textGridworldDisplay
    display = textGridworldDisplay.TextGridworldDisplay(world)
    if not opts.textDisplay:
      import graphicsGridworldDisplay
      display = graphicsGridworldDisplay.GraphicsGridworldDisplay(world, opts.gridSize, opts.speed)
    display.start()
  
    agents = []
    for index in range(0, opts.manual):
      agent = env.agentLabels[index]
      actionGetter = (lambda a: lambda state: env.getPossibleActions(state, a))(agent) 
      agents.append(UserAgent(actionGetter))
  
    for index in range(opts.manual, len(env.agentLabels)):
      agent = env.agentLabels[index]

      actionGetter = (lambda a: lambda state: env.getPossibleActions(state, a))(agent) 
      actionFn = actionGetter

      observationFn = flatteningObserver
      if opts.partial:
        observationFn = radiusOneObserverFactory(world, agent)

      qLearnOpts = {'gamma': opts.discount, 
                    'alpha': opts.learningRate, 
                    'epsilon': opts.epsilon,
                    'actionFn': actionFn,
                    'observationFn': observationFn}
      agents.append(qlearningAgents.QLearningAgent(**qLearnOpts))

    returns = []
    for episode in range(1, opts.episodes+1):
      
      displayCallback = lambda state, agent, label: None
      if opts.quiet == -1 or episode > opts.quiet:
        displayCallback = lambda state, agent, label: display.displayQValues(agent, label, currentState=state, message="CURRENT Q-VALUES FOR "+label)
  
      messageCallback = lambda x: None
      if opts.quiet == -1 or episode > opts.quiet:
        messageCallback = lambda x: printString(x)

      pauseCallback = lambda: None
      if opts.pause and (opts.quiet == -1 or episode > opts.quiet):
        pauseCallback = lambda: display.pause()

      if episode == opts.quiet + 1:
        for agent in agents:
          if 'epsilon' in dir(agent):
            agent.epsilon = 0
        world.setNoise(0)

      print "STARTING EPISODE %d" % (episode)
      returns.append(runEpisode(agents, env, opts.discount, displayCallback, messageCallback, pauseCallback, episode))

    # if opts.episodes > 0:
    #   print
    #   print "AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes)

    if opts.quiet == -1 or episode > opts.quiet:
      for index, agent in enumerate(agents):
        label = env.agentLabels[index]
        display.displayQValues(agent, env.agentLabels[index], message="Q-VALUES FOR "+label+" AFTER "+str(opts.episodes)+" EPISODES")
        display.pause()

    timeString = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    fileName = "results_n%0.2f_r%0.2f_k%d_%s_%s.txt" % (opts.noise, opts.livingReward, opts.episodes, opts.grid, timeString)

    f = open(fileName, 'w')
    f.write(str(env.agentLabels) + "\n")
    for episode, returnCounter in enumerate(returns):
      f.write("%d, %s\n" % (episode, ", ".join(str(returnCounter[label]) for label in env.agentLabels)))
    f.close()

  except SystemExit:
    pass
  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)