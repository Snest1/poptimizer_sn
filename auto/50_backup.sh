# optimize eternal briefcase by resample_sn
v_date_YYYYMMDD_HHmm=$(date "+%Y%m%d_%H%M")

echo "BATCH 'backup all' $v_date_YYYYMMDD_HHmm"

mongodump --out=/home/sn/sn/poptimizer-master/my_dumps/dump_${v_date_YYYYMMDD_HHmm}/mongo
mongoexport --db=data --collection=models --out=/home/sn/sn/poptimizer-master/my_dumps/dump_${v_date_YYYYMMDD_HHmm}/mongo/models_${v_date_YYYYMMDD_HHmm}.json

cp -aL /home/sn/sn/poptimizer-master/portfolio/. /home/sn/sn/poptimizer-master/my_dumps/dump_${v_date_YYYYMMDD_HHmm}/portfolio/
cp -aL /home/sn/sn/poptimizer-master/auto/. /home/sn/sn/poptimizer-master/my_dumps/dump_${v_date_YYYYMMDD_HHmm}/auto/


