v_date_YYYYMMDD_HHmm=$(date "+%Y%m%d_%H%M")

echo $(date "+%Y.%m.%d %H:%M:%S") "Manual starting stopper..."  >> evolve.log


# Loop forever (until break is issued)
while true; do
    echo $(date "+%Y.%m.%d %H:%M:%S") "Stopper"
    sleep 10
done
