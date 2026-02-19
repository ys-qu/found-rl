#!/bin/bash

export HYDRA_FULL_ERROR=1
export CARLA_ROOT=/workspace/CARLA
#export PYTHONPATH=$PYTHONPATH:/home/ai/qys/softwares/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg

# * To collect from Autopilot for the NoCrash benchmark
eval_ () {
  python -u eval_vlm.py resume=true log_video=false save_to_wandb=true \
  agent.rl_vlm.wb_run_path=null \
  wb_project=found_rl wb_name=internvl3_1b_full_sft_bev_lb_test\
  test_suites=lb_test\
  n_episodes=160 \
  carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
}


# NO NEED TO MODIFY THE FOLLOWING
# actiate conda env
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate found_rl

# remove checkpoint files
# rm outputs/checkpoint.txt
# rm outputs/wb_run_id.txt
# rm outputs/ep_stat_buffer_*.json

# resume benchmark in case carla is crashed.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  pkill -9 CarlaUE4 2>/dev/null
  pkill -9 CarlaUE4 2>/dev/null
  pkill -9 CarlaUE4 2>/dev/null
  pkill -9 python 2>/dev/null
  pkill -9 python 2>/dev/null
  pkill -9 python 2>/dev/null

  eval_
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

killall -9 -r CarlaUE4-Linux
echo "Bash script done."

# To shut down the aws instance after the script is finished
# sleep 10
# sudo shutdown -h now