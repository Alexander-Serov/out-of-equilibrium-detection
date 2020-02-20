#!/bin/bash

# This script efficiently copies calculated data from server to a local folder taking only files modifed after a given date
# For increased efficiency, the files are first tarred and then untarred
# Needs to be launched from a UNIX environment

NEWER_THAN="2019-10-29 00:00:00" # default start date
SERVER_PATH=./out-of-equilibrium-detection/job-manager/data/
LOCAL_PATH=/mnt/d/calculated_data/out-of-equilibrium_detection/
EXTENSION="*.pyc"
NUM_FILES=0

# Save current date & time for further re-use
ssh aserov@tars.pasteur.fr touch start_date2

# Check if last copy date file exists
if ssh aserov@tars.pasteur.fr "test -e" start_date;
then
        NEWER_THAN=`ssh aserov@tars.pasteur.fr "date -r" start_date`
else
        ssh aserov@tars.pasteur.fr touch --date "'$NEWER_THAN'" start_date
fi
echo "Started copy on $(date)."
echo "Copying results calculated after $NEWER_THAN."

# Check if any files need to be copied
#NUM_FILES=`ssh aserov@tars.pasteur.fr find $SERVER_PATH -type f -newer start_date -name "$EXTENSION" | wc -l`
#echo "Found $NUM_FILES files to copy..."
#if (($NUM_FILES>0))
#then
echo "Compressing..."
ssh aserov@tars.pasteur.fr rm -f ./copy.tar
# xform is used to drop the absolute path
ssh aserov@tars.pasteur.fr find $SERVER_PATH -type f -newer start_date -name "$EXTENSION" -exec "tar rf ./copy.tar --xform='s|.*/||' --show-transformed-names {} +"
echo "Done!"

# Copy
if ssh aserov@tars.pasteur.fr "test ./copy.tar";
then
    scp aserov@tars.pasteur.fr:./copy.tar ./

    NUM_FILES=`tar -tf ./copy.tar | wc -l`
    echo "$NUM_FILES files received from server"

    echo "Extracting..."
    tar xf ./copy.tar -C $LOCAL_PATH --overwrite
    echo "Done!"

    # Cleaning up
    rm ./copy.tar
    ssh aserov@tars.pasteur.fr rm ./copy.tar
else
    echo "No files to copy. Finished!"
fi

# Replace old-date file with a new one
if [ $NUM_FILES -gt 0 ];
then
    ssh aserov@tars.pasteur.fr mv -f start_date2 start_date
fi



#### ssh aserov@tars.pasteur.fr rm ./start_date