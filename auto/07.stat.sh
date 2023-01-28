# optimize eternal briefcase by resample_sn
v_date_YYYYMMDD_HHmm=$(date "+%Y%m%d_%H%M")

echo "BATCH 'stat' $v_date_YYYYMMDD_HHmm"

#mongodump --out=/home/sn/sn/poptimizer-master/my_dumps/dump_${v_date_YYYYMMDD_HHmm}/mongo
#mongoexport --db=data --collection=models --out=/home/sn/sn/poptimizer-master/my_dumps/dump_${v_date_YYYYMMDD_HHmm}/mongo/models_${v_date_YYYYMMDD_HHmm}.json

cd /home/sn/sn/poptimizer-master/

#python3 -m sn_stat.py > /home/sn/sn/poptimizer-master/stat_${v_date_YYYYMMDD_HHmm}_tmp
source ./.venv/bin/activate
python -m sn_stat.py > /home/sn/sn/poptimizer-master/stat_${v_date_YYYYMMDD_HHmm}_tmp
deactivate

cat /home/sn/sn/poptimizer-master/stat_${v_date_YYYYMMDD_HHmm}_tmp | tr '.' ',' > /home/sn/sn/poptimizer-master/stat_${v_date_YYYYMMDD_HHmm}
rm /home/sn/sn/poptimizer-master/stat_${v_date_YYYYMMDD_HHmm}_tmp


#cp -aL /home/sn/sn/poptimizer-master/portfolio/. /home/sn/sn/poptimizer-master/my_dumps/dump_${v_date_YYYYMMDD_HHmm}/portfolio/
#cp -aL /home/sn/sn/poptimizer-master/auto/. /home/sn/sn/poptimizer-master/my_dumps/dump_${v_date_YYYYMMDD_HHmm}/auto/


