#!/usr/bin/env python
from ants import *

# define a class with a do_turn method
# the Ants.run method will parse and update bot input
# it will also run the do_turn method for us
class MyBot:
    def __init__(self):
        # define class level variables, will be remembered between turns
        pass
    
    # do_setup is run once at the start of the game
    # after the bot has received the game settings
    # the ants class is created and setup by the Ants.run method
    def do_setup(self, ants):
        # initialize data structures after learning the game settings
        self.size_of_map = ants.rows * ants.cols
        self.hills = []
        self.unseen = []
        for row in range(ants.rows):
            for col in range(ants.cols):
                self.unseen.append((row, col))



    def features(self, ants):
        def min_distance(l1, l2):
            def min_d(loc, l):
                return min(ants.distance(loc, loc1) for loc1 in l)

            return min(min_d(loc, l2) for loc in l1)
        features = []
        features.append(len(ants.my_ants()))  # num of ants
        features.append(min_distance(ants.my_ants(), ants.food()))  # closest food
        features.append(len(ants.enemy_hills()))  # num of visible enemy hills
        features.append(len(ants.enemy_ants()))  # num of visible enemy ants
        features.append(0)  # num of ants to die
        features.append(0)  # num of ants we will kill
        features.append(min_distance(ants.enemy_ants(), ants.my_hills()))  # min dist of enemy ant to our hills
        # features.append(float(len(ants.visible())))  # % of map visible
        features.append(min_distance(ants.my_ants(), ants.my_hills()))  # min dist of our ants from our hills

        return

    # do turn is run once per turn
    # the ants class has the game state and is updated by the Ants.run method
    # it also has several helper methods to use
    def do_turn(self, ants):
        # loop through all my ants and try to give them orders
        # the ant_loc is an ant location tuple in (row, col) form
        orders = {}
        
        # prevent stepping on my own motherfucking hills
        for hill_loc in ants.my_hills():
            orders[hill_loc] = None
            
        def do_move_direction(location, direction):
            new_location = ants.destination(location, direction)
            if ants.unoccupied(new_location) and new_location not in orders:
                ants.issue_order((location, direction))
                orders[new_location] = location
                return True
            else:
                return False
        
        targets = {}
        def do_move_location(loc, dest):
            directions = ants.direction(loc, dest)
            for d in directions:
                if do_move_direction(loc, d):
                    targets[dest] = loc
                    return True
            return False
        
        # take food
        ant_dist = []
        for food_loc in ants.food():
            for ant_loc in ants.my_ants():
                distance = ants.distance(ant_loc, food_loc)
                ant_dist.append((distance, ant_loc, food_loc))
        ant_dist.sort()
        for dist, ant_loc, food_loc in ant_dist:
            if food_loc not in targets and ant_loc not in targets.values():
                do_move_location(ant_loc, food_loc)
        
        for hill_loc in ants.my_hills():
            if hill_loc in ants.my_ants() and hill_loc not in orders.values():
                for direction in ('s','e','w','n'):
                    if do_move_direction(hill_loc, direction):
                        break
                        
        # attack hills
        for hill_loc, hill_owner in ants.enemy_hills():
            if hill_loc not in self.hills:
                self.hills.append(hill_loc)        
        ant_dist = []
        for hill_loc in self.hills:
            for ant_loc in ants.my_ants():
                if ant_loc not in orders.values():
                    dist = ants.distance(ant_loc, hill_loc)
                    ant_dist.append((dist, ant_loc, hill_loc))
        ant_dist.sort()
        for dist, ant_loc, hill_loc in ant_dist:
            do_move_location(ant_loc, hill_loc)
                        
        # explore unseen areas
        for loc in self.unseen[:]:
            if ants.visible(loc):
                self.unseen.remove(loc)
        for ant_loc in ants.my_ants():
            if ant_loc not in orders.values():
                unseen_dist = []
                for unseen_loc in self.unseen:
                    dist = ants.distance(ant_loc, unseen_loc)
                    unseen_dist.append((dist, unseen_loc))
                unseen_dist.sort()
                for dist, unseen_loc in unseen_dist:
                    if do_move_location(ant_loc, unseen_loc):
                        break
        
if __name__ == '__main__':
    # psyco will speed up python a little, but is not needed
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    
    try:
        # if run is passed a class with a do_turn method, it will do the work
        # this is not needed, in which case you will need to write your own
        # parsing function and your own game state class
        Ants.run(MyBot())
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
