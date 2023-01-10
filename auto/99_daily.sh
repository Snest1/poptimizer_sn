# optimize eternal briefcase by resample_sn
v_date_YYYYMMDD_HHmm=$(date "+%Y%m%d_%H%M")

echo "BATCH 'daily' $v_date_YYYYMMDD_HHmm"

cp /home/sn/sn/poptimizer-master/auto/07.stat.sh /home/sn/sn/poptimizer-master/auto/!work/
cp /home/sn/sn/poptimizer-master/auto/50_backup.sh /home/sn/sn/poptimizer-master/auto/!work/


#cp /home/sn/sn/poptimizer-master/auto/10_opt_eternal.sh /home/sn/sn/poptimizer-master/auto/!work/
cp /home/sn/sn/poptimizer-master/auto/30_opt_my_ideal.sh /home/sn/sn/poptimizer-master/auto/!work/
#cp /home/sn/sn/poptimizer-master/auto/33_opt_for_1attempt.sh /home/sn/sn/poptimizer-master/auto/!work/
cp /home/sn/sn/poptimizer-master/auto/31_opt_my_idear_ystd.sh /home/sn/sn/poptimizer-master/auto/!work/

