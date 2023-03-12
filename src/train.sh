#!/usr/bin/env bash


train_data=../data/VAST/vast_train.csv
dev_data=../data/VAST/vast_dev.csv
test_data=../data/VAST/vast_test.csv

echo "training model with early stopping and $3 warm-up epochs"
python train_model.py -s $1 -i ${train_data} -d ${test_data} -e 1 -p $2 -k $3
