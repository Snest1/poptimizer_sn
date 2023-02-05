# optimize eternal briefcase by resample_sn
v_date_YYYYMMDD_HHmm=$(date "+%Y%m%d_%H%M")

dump_folder=/home/sn/sn/poptimizer-master/my_dumps/${v_date_YYYYMMDD_HHmm}_opt_my_ideal
port=02.my_cur_port.yaml
py_from=/home/sn/sn/poptimizer-master/auto/src/05_my_ideal/optimizer_resample_sn.py

config=/home/sn/sn/poptimizer-master/config/sn_config.resample_sn



now=$(date "+%Y%m%d %H%M%S")
echo "BATCH 'optimize eternal briefcase by resample_sn' $v_date_YYYYMMDD_HHmm started at $now"


mkdir ${dump_folder}/



mv /home/sn/sn/poptimizer-master/config/config.yaml /home/sn/sn/poptimizer-master/config/config.yaml_back
cp ${config} /home/sn/sn/poptimizer-master/config/config.yaml


cp /home/sn/sn/poptimizer-master/portfolio/sn/${port} /home/sn/sn/poptimizer-master/portfolio/${port}


cp /home/sn/sn/poptimizer-master/poptimizer/portfolio/optimizer_resample_sn.py /home/sn/sn/poptimizer-master/poptimizer/portfolio/optimizer_resample_sn.py_back
cp ${py_from} /home/sn/sn/poptimizer-master/poptimizer/portfolio/optimizer_resample_sn.py

cd /home/sn/sn/poptimizer-master/

#.venv/bin/python3 -m poptimizer optimize --for-sell 3 $(date "+%Y-%m-%d") > ${dump_folder}/log_${v_date_YYYYMMDD_HHmm}
##python3 -m poptimizer optimize --for-sell 3 $(date "+%Y-%m-%d") > ${dump_folder}/log_${v_date_YYYYMMDD_HHmm}
source /home/sn/sn/poptimizer-master/.venv/bin/activate
python -m poptimizer optimize --for-sell 3 $(date "+%Y-%m-%d") > ${dump_folder}/log_${v_date_YYYYMMDD_HHmm}
#python -m poptimizer optimize --for-sell 3 $(date "+%Y-%m-%d")
deactivate
#.venv/bin/python3 -m poptimizer optimize --for-sell 3 $(date "+%Y-%m-%d") > ${dump_folder}/log_${v_date_YYYYMMDD_HHmm}
##python3 -m poptimizer optimize --for-sell 3 $(date "+%Y-%m-%d") > ${dump_folder}/log_${v_date_YYYYMMDD_HHmm}


cp /home/sn/sn/poptimizer-master/poptimizer/portfolio/optimizer_resample_sn.py ${dump_folder}/optimizer_resample_sn.py
mv /home/sn/sn/poptimizer-master/poptimizer/portfolio/optimizer_resample_sn.py_back /home/sn/sn/poptimizer-master/poptimizer/portfolio/optimizer_resample_sn.py
mv /home/sn/sn/poptimizer-master/portfolio/${port} ${dump_folder}/!${v_date_YYYYMMDD_HHmm}_${port}_before.yaml
mv /home/sn/sn/poptimizer-master/out.txt ${dump_folder}/out_${v_date_YYYYMMDD_HHmm}.txt
mv /home/sn/sn/poptimizer-master/my_dumps/ops*.xlsx ${dump_folder}/ops_${v_date_YYYYMMDD_HHmm}.xlsx


mv /home/sn/sn/poptimizer-master/config/config.yaml_back /home/sn/sn/poptimizer-master/config/config.yaml


now=$(date "+%Y%m%d %H%M%S")
echo "BATCH 'optimize eternal briefcase by resample_sn' $v_date_YYYYMMDD_HHmm ended at $now"

sleep 30










#PKG_MNG=brew
#PYTHON_VER=python@3.10
#TOOLS=${PYTHON_VER} poetry mongodb-community mongodb-database-tools
#SRC=poptimizer
#VENV_NAME=.venv
#VENV_ACTIVATE=. ${VENV_NAME}/bin/activate
#PYTHON=./../.venv/bin/python3

#$(VENV_ACTIVATE)
#$(PYTHON) -m $(SRC) evolve
