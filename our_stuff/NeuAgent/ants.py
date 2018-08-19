#!/usr/bin/env python
import sys
import traceback
import random
import time
from collections import defaultdict
from math import sqrt
import numpy as np

MY_ANT = 0
ANTS = 0
DEAD = -1
LAND = -2
FOOD = -3
WATER = -4

# MAP MATRIX VALUES
M_SELF = 0
M_LAND = np.array([1, 0, 0, 0, 0])
M_WATER = np.array([0, 1, 0, 0, 0])
M_ENEMY = np.array([0, 0, 1, 0, 0])
M_FRIEND = np.array([0, 0, 0, 1, 0])
M_FOOD = np.array([0, 0, 0, 0, 1])

PLAYER_ANT = 'abcdefghij'
HILL_ANT = string = 'ABCDEFGHIJ'
PLAYER_HILL = string = '0123456789'
MAP_OBJECT = '?%*.!'
MAP_RENDER = PLAYER_ANT + HILL_ANT + PLAYER_HILL + MAP_OBJECT
NOOP = "No-Op"

OBSERVABILITY_SQUARE_SIZE = 17

AIM = {'n': (-1, 0),
       'e': (0, 1),
       's': (1, 0),
       'w': (0, -1)}
RIGHT = {'n': 'e',
         'e': 's',
         's': 'w',
         'w': 'n'}
LEFT = {'n': 'w',
        'e': 'n',
        's': 'e',
        'w': 's'}
BEHIND = {'n': 's',
          's': 'n',
          'e': 'w',
          'w': 'e'}

class Ants():
    def __init__(self):
        self.cols = None
        self.rows = None
        self.map = None
        self.hill_list = {}
        self.ant_list = {}
        self.dead_list = defaultdict(list)
        self.food_list = []
        self.turntime = 0
        self.loadtime = 0
        self.turn_start_time = None
        self.vision = None
        self.viewradius2 = 0
        self.attackradius2 = 0
        self.spawnradius2 = 0
        self.turns = 0
        self.turns_so_far = 0
        self.orders = {}
        self.orders2 = {}
        self.indices_in_attack_range = set()
        self.edge_of_view = []

    def setup(self, data):
        'parse initial input and setup starting game state'
        for line in data.split('\n'):
            line = line.strip().lower()
            if len(line) > 0:
                tokens = line.split()
                key = tokens[0]
                if key == 'cols':
                    self.cols = int(tokens[1])
                elif key == 'rows':
                    self.rows = int(tokens[1])
                elif key == 'player_seed':
                    random.seed(int(tokens[1]))
                elif key == 'turntime':
                    self.turntime = int(tokens[1])
                elif key == 'loadtime':
                    self.loadtime = int(tokens[1])
                elif key == 'viewradius2':
                    self.viewradius2 = int(tokens[1])
                elif key == 'attackradius2':
                    self.attackradius2 = int(tokens[1])
                elif key == 'spawnradius2':
                    self.spawnradius2 = int(tokens[1])
                elif key == 'turns':
                    self.turns = int(tokens[1])
        self.map = [[LAND for col in range(self.cols)]
                    for row in range(self.rows)]
        self.map_mat = np.full((self.rows, self.cols, len(M_LAND)), M_LAND)
        self.calc_attack_range_matrix()
        self.calc_edge_of_view()
        self.turns_so_far = 0

    def neighbourhood_offsets(self, max_dist, full=True):
        """ Return a list of squares within a given distance of loc

            Loc is not included in the list
            For all squares returned: 0 < distance(loc,square) <= max_dist

            Offsets are calculated so that:
              -height <= row+offset_row < height (and similarly for col)
              negative indicies on self.map wrap thanks to python
        """
        offsets = []
        mx = int(sqrt(max_dist))
        for d_row in range(-mx,mx+1):
            for d_col in range(-mx,mx+1):
                d = d_row**2 + d_col**2
                if full:
                    if 0 < d <= max_dist:
                        offsets.append((
                            d_row%self.rows-self.rows,
                            d_col%self.cols-self.cols
                        ))
                else:
                    if max_dist -1 < d <= max_dist:
                        offsets.append((
                            d_row%self.rows-self.rows,
                            d_col%self.cols-self.cols
                        ))
        return offsets

    def calc_edge_of_view(self):
        'precalculate edge of view circle'
        self.edge_of_view = self.neighbourhood_offsets(self.viewradius2, False)


    def calc_attack_range_matrix(self):
        'precalculate possible relative indices in attack range'
        # attack_range = self.attackradius2**0.5
        # for i in range(int(-attack_range), int(attack_range)):
        #     for j in range(int(-attack_range), int(attack_range)):
        #         if i**2 + j**2 < self.attackradius2:
        #             self.indices_in_attack_range.add((i,j))
        self.indices_in_attack_range = self.neighbourhood_offsets(self.attackradius2)

    def update(self, data):
        'parse engine input and update the game state'
        # start timer
        self.turn_start_time = time.time()
        
        # reset vision
        self.vision = None
        
        # clear hill, ant and food data
        self.hill_list = {}
        for row, col in self.ant_list.keys():
            self.map[row][col] = LAND
            self.map_mat[row, col] = M_LAND
        self.ant_list = {}
        for row, col in self.dead_list.keys():
            self.map[row][col] = LAND
            self.map_mat[row, col] = M_LAND
        self.dead_list = defaultdict(list)
        for row, col in self.food_list:
            self.map[row][col] = LAND
            self.map_mat[row, col] = M_LAND
        self.food_list = []
        
        # update map and create new ant and food lists
        for line in data.split('\n'):
            line = line.strip().lower()
            if len(line) > 0:
                tokens = line.split()
                if len(tokens) >= 3:
                    row = int(tokens[1])
                    col = int(tokens[2])
                    if tokens[0] == 'w':
                        self.map[row][col] = WATER
                        self.map_mat[row, col] = M_WATER
                    elif tokens[0] == 'f':
                        self.map[row][col] = FOOD
                        self.map_mat[row, col] = M_FOOD
                        self.food_list.append((row, col))
                    else:
                        owner = int(tokens[3])
                        m_ant = M_FRIEND if owner == 'a' else M_ENEMY
                        if tokens[0] == 'a':
                            self.map[row][col] = owner
                            self.map_mat[row, col] = m_ant
                            self.ant_list[(row, col)] = owner
                        elif tokens[0] == 'd':
                            # food could spawn on a spot where an ant just died
                            # don't overwrite the space unless it is land
                            if self.map[row][col] == LAND:
                                self.map[row][col] = DEAD
                            # but always add to the dead list
                            self.dead_list[(row, col)].append(owner)
                        elif tokens[0] == 'h':
                            owner = int(tokens[3])
                            self.hill_list[(row, col)] = owner
                        
    def time_remaining(self):
        return self.turntime - int(1000 * (time.clock() - self.turn_start_time))
    
    def issue_order(self, order):
        'issue an order by writing the proper ant location and direction'
        (row, col), direction = order
        self.orders2[(row, col)] = direction
        if direction != NOOP:
            sys.stdout.write('o %s %s %s\n' % (row, col, direction))
            sys.stdout.flush()
        
    def finish_turn(self):
        'finish the turn by writing the go line'
        self.orders = {}
        self.orders2 = {}
        self.turns_so_far += 1
        sys.stdout.write('go\n')
        sys.stdout.flush()

    def my_hills(self):
        return [loc for loc, owner in self.hill_list.items()
                    if owner == MY_ANT]

    def enemy_hills(self):
        return [(loc, owner) for loc, owner in self.hill_list.items()
                    if owner != MY_ANT]
        
    def my_ants(self):
        'return a list of all my ants'
        return [(row, col) for (row, col), owner in self.ant_list.items()
                    if owner == MY_ANT]

    def enemy_ants(self):
        'return a list of all visible enemy ants'
        return [((row, col), owner)
                    for (row, col), owner in self.ant_list.items()
                    if owner != MY_ANT]

    def food(self):
        'return a list of all food locations'
        return self.food_list[:]

    def passable(self, loc):
        'true if not water'
        row, col = loc
        return self.map[row][col] > WATER
    
    def unoccupied(self, loc):
        'true if no ants are at the location'
        row, col = loc
        return self.map[row][col] in (LAND, DEAD)

    def unoccupied_including_orders(self, loc):
        'similar to unoccupied, except taking into consideration other ants order'
        row, col = loc
        if not self.passable((row, col)):
            return False
        if self.map[row][col] == FOOD:
            return False
        if self.map[row][col] == MY_ANT:
            return False
        if (row, col) in self.orders:
            return False
        return True


    def destination(self, loc, direction):
        'calculate a new location given the direction and wrap correctly'
        if direction is NOOP: # TODO: for now None signifies no action. Open for reconsideration.
            return loc
        row, col = loc
        d_row, d_col = AIM[direction]
        return ((row + d_row) % self.rows, (col + d_col) % self.cols)

    def wrap(self, loc):
        'wrap coordinates'
        row, col = loc
        return row % self.rows, col % self.cols

    def destination_with_obstacles(self, loc, direction):
        'add a test for passability of new location and return old location if not'
        new_loc = self.destination(loc, direction)
        if not self.passable(new_loc): # TODO: Add test for food?
            return loc
        return new_loc

    def distance_manhattan(self, loc1, loc2):
        'calculate the closest distance between to locations'
        row1, col1 = loc1
        row2, col2 = loc2
        d_col = min(abs(col1 - col2), self.cols - abs(col1 - col2))
        d_row = min(abs(row1 - row2), self.rows - abs(row1 - row2))
        return d_row + d_col


    def direction(self, loc1, loc2):
        'determine the 1 or 2 fastest (closest) directions to reach a location'
        row1, col1 = loc1
        row2, col2 = loc2
        height2 = self.rows//2
        width2 = self.cols//2
        d = []
        if row1 < row2:
            if row2 - row1 >= height2:
                d.append('n')
            if row2 - row1 <= height2:
                d.append('s')
        if row2 < row1:
            if row1 - row2 >= height2:
                d.append('s')
            if row1 - row2 <= height2:
                d.append('n')
        if col1 < col2:
            if col2 - col1 >= width2:
                d.append('w')
            if col2 - col1 <= width2:
                d.append('e')
        if col2 < col1:
            if col1 - col2 >= width2:
                d.append('e')
            if col1 - col2 <= width2:
                d.append('w')
        return d

    def visible(self, loc):
        ' determine which squares are visible to the given player '

        if self.vision == None:
            if not hasattr(self, 'vision_offsets_2'):
                # precalculate squares around an ant to set as visible
                self.vision_offsets_2 = []
                mx = int(sqrt(self.viewradius2))
                for d_row in range(-mx,mx+1):
                    for d_col in range(-mx,mx+1):
                        d = d_row**2 + d_col**2
                        if d <= self.viewradius2:
                            self.vision_offsets_2.append((
                                d_row%self.rows-self.rows,
                                d_col%self.cols-self.cols
                            ))
            # set all spaces as not visible
            # loop through ants and set all squares around ant as visible
            self.vision = [[False]*self.cols for row in range(self.rows)]
            for ant in self.my_ants():
                a_row, a_col = ant
                for v_row, v_col in self.vision_offsets_2:
                    self.vision[a_row+v_row][a_col+v_col] = True
        row, col = loc
        return self.vision[row][col]

    def num_of_revealed_on_edge(self, ant_loc, tile):
        'returns a list of locations on the rim viewradius of and ant at loc'
        row, col = ant_loc
        for (d_row, d_col) in self.edge_of_view:
            edge_tile = self.wrap((row + d_row, col + d_col))
            # if :
            #     return True
        return False

    def attack_range_of_loc(self, loc, prev_ants):
        'return list of allies, enemies and dead enemies'
        row, col = loc
        friendly_ants = []
        live_enemy_ants = []
        dead_enemy_ants = []
        for (i,j) in self.indices_in_attack_range:
            # tile = self.wrap((row + i, col + j))
            # if tile in self.my_ants():
            #     friendly_ants.append(tile)
            # elif tile in self.enemy_ants():
            #     live_enemy_ants.append(tile)
            # elif tile in self.dead_list:
            #     dead = set(self.dead_list)
            #     if dead and dead != {0}:
            #         dead_enemy_ants.append(tile)

            tile = self.map[row + i][col + j]
            tile_loc = self.wrap((row+i, col+j))
            if tile == 0:
                friendly_ants.append(tile_loc)
            elif tile > 0:
                live_enemy_ants.append(tile_loc)
            elif tile == DEAD and 0 not in self.dead_list[tile_loc]:
                dead_enemy_ants.append(tile_loc)

        return friendly_ants, live_enemy_ants, dead_enemy_ants

    def remember_order(self, loc, direction):
        'add order to order dict'
        new_loc = self.destination_with_obstacles(loc, direction)
        self.orders[new_loc] = loc

    def render_text_map(self):
        'return a pretty string representing the map'
        tmp = ''
        for row in self.map:
            tmp += '# %s\n' % ''.join([MAP_RENDER[col] for col in row])
        return tmp

    def ant_observation(self, loc): # TODO: verify that it is working correctly
        side = OBSERVABILITY_SQUARE_SIZE / 2
        rows = [i % self.rows for i in range(loc[0]-side, loc[0]+side+1)]
        cols = [i % self.cols for i in range(loc[1] - side, loc[1] + side+1)]
        return self.map_mat[np.ix_(rows, cols)]

    def __eq__(self, other):
        return self.map == other.map

