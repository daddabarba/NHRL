A small and simple group of scripts that should facilitate using python modules
nested in multiple directories.

## How to Run
Simply run `source ./addPaths.sh` and all the sub directories of the parent one
(of the repository) will be added to `PYTHONPATH`. <br />
It is also possible to specify which folder should be (recursively) added to `PYTHONPATH`, by specifying a relative path as argument (eg. `source ./addPaths.sh ../../`). Thus `source ./addPaths.sh` is equivalent to running `source ./addPaths.sh ..`.

The environment variable `PYTHONPATH` should be reset once the current process is closed, however you can manually reset it by running `source ./removePaths.sh`
. For more safety, you can run the function `stripAdditions` in the `getModules.py` module, and then find the original `PYTHONPATH` value in `<repository>/data/stripped_paths`.
