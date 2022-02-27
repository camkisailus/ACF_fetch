
Step 1: Open 5 seperate terminals

Step 2: In 1 terminal `ssh` into fetch and run `source Downloads/hjz_ws/devel/setup.sh` and `roslaunch fetch_config fetch_config.launch`

Step 3: In a different terminal, run `fetch 18` then `cd ~/Desktop/py_move_group_ws` and `source devel/setup.zsh` and `rosrun py_move_group move_group_node.py`. This is the node that gets requests from `~/Desktop/ACF_fetch/src/driver/src/main.py` and sends requests to the `fetch_config` node for planning.

Step 4: In a different terminal, run `fetch 18` then `source ~/catkin_build_ws/install/setup.zsh --extend` and `source ~/Desktop/ACF_fetch/devel/setup.zsh --extend`. Then `cd ~/Desktop/ACF_fetch` and run `python3 src/driver/src/main.py`. This is the script that takes user input for what object to grasp.

Step 5: In a different terminal, run `fetch 18` then `source ~/catkin_build_ws/install/setup.zsh --extend` and `source ~/Desktop/ACF_fetch/devel/setup.zsh --extend`. Then `cd ~/Desktop/ACF_fetch` and run `roslaunch acf_network acf_network.launch`. This is the ACF network node. It just runs detection when the user requests them from `~/Desktop/ACF_fetch/src/driver/src/main.py`

Step 6: In the last terminal, run `fetch 18` and `rviz -d ~/Desktop/ACF_fetch/acf_fetch.rviz`