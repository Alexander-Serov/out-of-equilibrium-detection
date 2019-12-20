#!/bin/bash

# Copy files to cluster for a new code execution
# Note that spaces in names are treated differently in the 2 cases

scp "/mnt/d/Google Drive/git/out-of-equilibrium-detection/arguments.dat" aserov@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/
scp "/mnt/d/Google Drive/git/out-of-equilibrium-detection/position.dat" aserov@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/
for file in /mnt/d/Google\ Drive/git/out-of-equilibrium-detection/*.py
do
    scp "$file" aserov@tars.pasteur.fr:./out-of-equilibrium-detection/
done

 # scp ./job-manager/*.py aserov@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/
 # scp ./job-manager/*.sh aserov@tars.pasteur.fr:./out-of-equilibrium-detection/job-manager/