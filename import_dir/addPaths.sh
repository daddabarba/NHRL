prev=..
python getModules.py ${1:-$prev}
additions=$(<data/sub_paths)
export PYTHONPATH=$PYTHONPATH$additions