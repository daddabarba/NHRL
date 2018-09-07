import sys
import os

from parameters import *

def generateAdditions(rel=""):
	if not os.path.isdir(FOLDER):
		os.makedirs(FOLDER)

	cwd = os.getcwd()
	wd = os.path.abspath(cwd + SYS_SEPARATOR + rel)

	with open(ADDED_SUB, "w") as f:
		for path, directories, files in os.walk(wd):
			if not path.split(SYS_SEPARATOR)[-1] in INGORE_LIST and not path.startswith(cwd):
				print(path)	
				f.write(BASH_SEPARATOR+path)
		f.write("\n")

def stripAdditions():

	with open(ADDED_SUB, "r") as f:
		addedDirs = f.read()
	addedDirs = addedDirs.split(":")

	currentDirs = sys.path

	for dir in addedDirs:
		if dir in currentDirs:
			del currentDirs[currentDirs.index(dir)]

	with open(STRIPPED_SUB, "w") as f:
		f.write(BASH_SEPARATOR.join(currentDirs))
		f.write("\n")

if __name__ == '__main__':
	generateAdditions(sys.argv[1]) if len(sys.argv)>1 else generateAdditions()
