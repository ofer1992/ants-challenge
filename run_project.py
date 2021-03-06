import os
import sys

HELP = 'run_project.py [arg]\n' \
       'arguments:\n' \
       'train - train vs HunterBot for a 100 rounds\n' \
       'fight - 1v1 battle with HunterBot\n' \
       'food - food collection contest\n' \
       'maze - maze food collection contest (uses a* so takes a while)\n' \
       'ffa - 4 bots free-for-all match'

def train():
    """train against hunterBot for a 100 rounds"""
    run_line = 'python tools/playgame_rl.py -v --nolaunch -e --turntime=10000 --log_dir game_logs --turns 300 --rounds 100 --map_file=our_stuff/maps/tutorial1.map "python our_stuff/approxQ_bot/AQbot.py" "python tools/sample_bots/python/HunterBot.py"'
    os.system(run_line)

def food_demo():
    """food collection contest"""
    run_line = 'python tools/playgame_rl.py -v -e --turntime=10000 --log_dir game_logs --turns 300 --rounds 1 --map_file=our_stuff/maps/food.map "python our_stuff/approxQ_bot/AQbot.py" "python tools/sample_bots/python/HunterBot.py" "python tools/sample_bots/python/GreedyBot.py" "python tools/sample_bots/python/RandomBot.py"'
    os.system(run_line)

def one_on_one():
    """1v1 battle between ApproxQBot and HunterBot"""
    run_line = 'python tools/playgame_rl.py -v -e --turntime=10000 --log_dir game_logs --turns 300 --rounds 1 --map_file=our_stuff/maps/tutorial1.map "python our_stuff/approxQ_bot/AQbot.py" "python tools/sample_bots/python/HunterBot.py"'
    os.system(run_line)

def maze():
    """food collection in maze"""
    run_line = 'python tools/playgame_rl.py -v -e --turntime=100000 --log_dir game_logs --turns 300 --rounds 1 --map_file=our_stuff/maps/food_maze.map "python our_stuff/approxQ_bot/AQbot_astar.py" "python tools/sample_bots/python/HunterBot.py" "python tools/sample_bots/python/GreedyBot.py" "python tools/sample_bots/python/RandomBot.py"'
    os.system(run_line)

def ffa():
    run_line = 'python tools/playgame_rl.py -v -e --turntime=10000 --log_dir game_logs --turns 300 --rounds 1 --map_file=our_stuff/maps/ffa.map "python our_stuff/approxQ_bot/AQbot.py" "python tools/sample_bots/python/HunterBot.py" "python tools/sample_bots/python/HunterBot.py" "python tools/sample_bots/python/HunterBot.py"'
    os.system(run_line)

if __name__ == '__main__':
    if len(sys.argv) <= 1 or sys.argv[1].lower() == "help":
        print HELP
        sys.exit(-1)
    args = sys.argv[1:]
    if args[0].lower() == "train":
        train()
    elif args[0].lower() == "fight":
        one_on_one()
    elif args[0].lower() == "food":
        food_demo()
    elif args[0].lower() == "maze":
        maze()
    elif args[0].lower() == "ffa":
        ffa()
    else:
        print HELP
        sys.exit(-1)
