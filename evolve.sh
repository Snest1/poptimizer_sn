v_date_YYYYMMDD_HHmm=$(date "+%Y%m%d_%H%M")

cd /home/sn/sn/poptimizer-master/
echo $(date "+%Y.%m.%d %H:%M:%S") "Manual starting..."  >> evolve.log


# Loop forever (until break is issued)
while true; do

    python3 -m poptimizer evolve
#    /home/sn/sn/poptimizer-master/daily.sh
#    cp /home/sn/sn/poptimizer-master/logs/lgr.lgr /home/sn/sn/poptimizer-master/my_dumps/lgr_${v_date_YYYYMMDD_HHmm}.lgr

    sleep 60
    echo $(date "+%Y.%m.%d %H:%M:%S") "Restarting..."  >> evolve.log
#    echo $(/bin/date)
#    echo "Restarting..." >> evolve.log

#    # Do a simple test for Internet connectivity
#    PacketLoss=$(ping "$TestIP" -c 2 | grep -Eo "[0-9]+% packet loss" | grep -Eo "^[0-9]")

#    # Exit the loop if ping is no longer dropping packets
#    if [ "$PacketLoss" == 0 ]; then
#        echo "Connection restored"
#        break
#    else
#        echo "No connectivity"
#    fi
done
