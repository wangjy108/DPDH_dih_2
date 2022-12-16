#!/bin/sh

###this the test line
export tool_home=/data1/zhengdan/jywang/tool_file/qm/bi_dih
#export PATH=/home/wangjiayue/tool_file/public/qm:$PATH

define_prefix=${1}
#initial_angle=${2}

cp ${tool_home}/rig_scan_prepare.py ./
cp ${tool_home}/rig_scan_result_bi.py ./
#cp ${tool_home}/rigid_param ./

cp ${tool_home}/run_gentor ./
cp ${tool_home}/gentor ./

#prepare.py
#result.py
#ini
#cp ${tool_home}/pyscf_opt_header.py ./
#cp ${tool_home}/pyscf_opt_geomparam.py ./

echo "prepare file for optimization run"
python rig_scan_prepare.py --mode opt --prefix ${define_prefix} --ini rigid_param
sleep 10s

echo "run optimization"
## submit for g16 calculation
g16 opt_${define_prefix}.gjf &
sleep 30s

for ((i=1;i++;i<=60))
do
    run_done_flag=`grep "Job cpu time" opt_${define_prefix}.log | wc -l`
    if [ "$run_done_flag" = "1" ];then
      echo "Finish optimize ${define_prefix}"
      break
    else
      sleep 30s
    fi
done

run_done_flag=`grep "Job cpu time" opt_${define_prefix}.log | wc -l`

if [ "$run_done_flag" = "1" ];then
  echo "prepare file for rigid scan"
  python rig_scan_prepare.py --mode scan --prefix ${define_prefix} --ini rigid_param
  gjf=`ls rigid_*.gjf`
  echo "run rigid scan"
  for file in $gjf
  do
    file_name=`basename $file .gjf`
    echo "scan ${file_name}"
    g16 $file &
    sleep 30s
    for ((i=1;i++;i<=60))
    do
      sp_flag=`grep "Job cpu time" ${file_name}.log | wc -l`
      if [ "$sp_flag" = "1" ];then
        echo "Finish ${file_name}"
        break
      else
        sleep 30s
      fi
    done
  done
else
  echo "check and ask for help"
  exit
fi

echo "Processing final results..."
python rig_scan_result_bi.py --prefix ${define_prefix} --ini rigid_param
mkdir PROCESS
mv *.log PROCESS
mv *.gjf PROCESS
rm -f *xyz
rm -f xyz*
echo "All Done"
