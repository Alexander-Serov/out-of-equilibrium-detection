#!/bin/bash

if [ -z "$1" ]
then
      echo "TARS username must be provided as an argument"
      end
else
      username=$1
fi


# Copy files to cluster for a new code execution
# Note that spaces in names are treated differently in the 2 cases

scp "./arguments.dat" ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/
#scp "./position.dat" ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/

for file in ./*.py
do
    scp "$file" ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/
done

scp "./requirements.txt" ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/
scp "./job-manager/start_me.py" ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/
scp "./job-manager/sbatch_tars.sh" ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/
scp "./job-manager/sbatch_tars_common.sh" ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/
scp "./job-manager/job_manager.py" ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/

 # scp ./job-manager/*.py ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/
 # scp ./job-manager/*.sh ${username}@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/