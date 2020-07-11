#!/bin/bash
squeue -u $USER
squeue -u $USER | grep -c R
squeue -u $USER | grep PD
