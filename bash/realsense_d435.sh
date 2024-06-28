source bash/init_dvrk.sh
roslaunch realsense2_camera rs_camera.launch json_file_path:=${PWD}/config/realsense_d435.json align_depth:=true filters:=spatial,temporal,decimation,disparity