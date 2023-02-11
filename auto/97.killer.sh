v_date_YYYYMMDD_HHmm=$(date "+%Y%m%d_%H%M")

echo $(date "+%Y.%m.%d %H:%M:%S") "Manual starting stopper..."  >> evolve.log


rm /home/sn/sn/poptimizer-master/auto/!work/97.killer.sh
pkill python
