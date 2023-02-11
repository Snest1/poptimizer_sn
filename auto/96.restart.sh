v_date_YYYYMMDD_HHmm=$(date "+%Y%m%d_%H%M")

echo $(date "+%Y.%m.%d %H:%M:%S") "Restarter start..."

rm /home/sn/sn/poptimizer-master/auto/!work/96.restart.sh
pkill python

cd /home/sn/sn/poptimizer-master/
echo $(date "+%Y.%m.%d %H:%M:%S") "Manual starting..."
python3 -m poptimizer evolve
