source bash/init_dvrk.sh
qlacloserelays
roslaunch dvrk_robot dvrk_arm_rviz.launch arm:=PSM1 config:=${PWD}/ext/dvrk_2_1/src/cisst-saw/sawIntuitiveResearchKit/share/cuhk-daVinci-2-0/console-PSM1.json # real robot