import sys, os

HELP = 'run_project.py [arg]\n' \
       'arguments:\n' \
       'train - train vs HunterBot for a 100 rounds\n' \
       'fight - 1v1 battle with HunterBot\n' \
       'food - food collection contest\n' \
       'maze - maze food collection contest'

def train():
    """train against hunterBot for a 100 rounds"""
    run_line = 'python tools/playgame_rl.py --nolaunch -e --turntime=10000 --log_dir game_logs --turns 300 --rounds 100 --map_file=our_stuff/maps/tutorial1.map "python our_stuff/approxQ_bot_ofer/AQbot.py" "python tools/sample_bots/python/HunterBot.py"'
    os.system(run_line)

def food_demo():
    """food collection contest"""
    run_line = 'python tools/playgame_rl.py -e --turntime=10000 --log_dir game_logs --turns 300 --rounds 1 --map_file=our_stuff/maps/tutorial1.map "python our_stuff/approxQ_bot_ofer/AQbot.py" "python tools/sample_bots/python/HunterBot.py"'
    os.system(run_line)

def one_on_one():
    """1v1 battle between ApproxQBot and HunterBot"""
    run_line = 'python tools/playgame_rl.py -e --turntime=10000 --log_dir game_logs --turns 300 --rounds 1 --map_file=our_stuff/maps/tutorial1.map "python our_stuff/approxQ_bot_ofer/AQbot.py" "python tools/sample_bots/python/HunterBot.py"'
    os.system(run_line)

def maze():
    """food collection in maze"""
    run_line = 'python tools/playgame_rl.py -e --turntime=10000 --log_dir game_logs --turns 300 --rounds 1 --map_file=our_stuff/maps/tutorial1.map "python our_stuff/approxQ_bot_ofer/AQbot.py" "python tools/sample_bots/python/HunterBot.py"'
    os.system(run_line)


if __name__ == '__main__':
    if len(sys.argv) <= 1 or sys.argv[2].lower() == "help":
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
    else:
        print HELP
        sys.exit(-1)
