#!/bin/bash
export HYDRA_FULL_ERROR=1
export CARLA_ROOT=/workspace/CARLA
#export CUDA_LAUNCH_BLOCKING=1

# agent.rl_vlm.training.kwargs.explore_coef=0.05 \
#train_rl () {
#  python -u train_rl_asycn_vlm.py \
#  agent.rl_vlm.wb_run_path=null \
#  wb_project=found_rl wb_name=td3_noVlm_noWm_noR_eudata_bevmasks15-96_bs1e5_tt1e6_noStatic_LiteXtMaCNN \
#  agent/rl_vlm/training=td3_vlm \
#  carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
#}

# sac
# agent/rl_vlm/policy=sac_xtma_gaus \
# train_rl () {
#   python -u train_rl_asycn_vlm.py \
#   agent.rl_vlm.wb_run_path=null \
#   wb_project=found_rl wb_name=drqv2_Vlm_noWm_noR_lbdata_bevmasks15-96_bs1e5_tt1e6_noStatic_LiteXtMaCNN_openclipBin_rs1 \
#   agent/rl_vlm/training=drqv2_vlm \
#   carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
# }

# td3
#  agent/rl_vlm/policy=td3_xtma \
# train_rl () {
#   python -u train_rl_asycn_vlm.py \
#   agent.rl_vlm.wb_run_path=null \
#   wb_project=found_rl wb_name=td3_Vlm_noWm_noR_lbdata_bevmasks15-96_bs1e5_tt1e6_noStatic_LiteXtMaCNN_awac \
#   agent/rl_vlm/training=td3_vlm \
#   carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
# }

# ddpg
#train_rl () {
#  python -u train_rl_asycn_vlm.py \
#  agent.rl_vlm.wb_run_path=null \
#  wb_project=found_rl wb_name=ddpg_noVlm_noWm_lbdata_bevmasks15-96_bs1e5_tt1e6_noStatic_LiteXtMaCNN \
#  agent/rl_vlm/training=ddpg_vlm \
#  carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
#}

# drqv2
# agent/rl_vlm/policy=drqv2_xtma \
train_rl () {
python -u train_rl_asycn_vlm.py \
agent.rl_vlm.wb_run_path=null \
wb_project=found_rl wb_name=drqv2_lbdata_rClip \
agent/rl_vlm/training=drqv2_vlm \
carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
}


# actiate conda env
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate found_rl

# resume benchmark in case carla is crashed.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  echo "${RED}[INFO] Cleaning up residual processes...${NC}"
  pkill -9 CarlaUE4 2>/dev/null
  pkill -9 CarlaUE4 2>/dev/null
  pkill -9 CarlaUE4 2>/dev/null
  pkill -9 python 2>/dev/null
  pkill -9 python 2>/dev/null
  pkill -9 python 2>/dev/null

  train_rl
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

killall -9 -r CarlaUE4-Linux
echo "Bash script done."

# To shut down the aws instance after the script is finished
# sleep 10
# sudo shutdown -h now