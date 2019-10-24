#!/bin/bash

# This script efficiently copies calculated data from server to a local folder taking only files modifed after a given date
# For increased efficiency, the files are first tarred and then untarred
# Needs to be launched from a UNIX environment

NEWER_THAN=2019-10-24 # including the date
SERVER_PATH=./out-of-equilibrium-detection/job-manager/data/
LOCAL_PATH=/mnt/d/calculated_data/out-of-equilibrium_detection/
EXTENSION="*.pyc"

ssh aserov@tars.pasteur.fr touch --date $NEWER_THAN start_date
echo "Found" `ssh aserov@tars.pasteur.fr find $SERVER_PATH -type f -newer start_date -name "$EXTENSION" | wc -l` "files to copy"

ssh aserov@tars.pasteur.fr rm -f ./copy.tar
echo "Compressing..."
ssh aserov@tars.pasteur.fr find $SERVER_PATH -type f -newer start_date -name "$EXTENSION" -exec "tar rf ./copy.tar --xform='s|.*/||' --show-transformed-names {} +"

# xform is used to drop the absolute path
echo "Done!"
scp aserov@tars.pasteur.fr:./copy.tar ./
echo "Extracting..."
tar xf ./copy.tar -C $LOCAL_PATH --overwrite
echo "Done!"

# Cleaning up
rm ./copy.tar
ssh aserov@tars.pasteur.fr rm ./copy.tar

# ssh aserov@tars.pasteur.fr rm ./start_date