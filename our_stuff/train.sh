#!/usr/bin/env sh
python ../tools/playgame.py "python RLbot.py train" "python ../tools/sample_bots/python/HunterBot.py" --map_file simpler.map --log_dir ../game_logs --turns 60 --food none --rounds=10 --turns 60 --verbose -e --nolaunch --scenario --turntime 3000