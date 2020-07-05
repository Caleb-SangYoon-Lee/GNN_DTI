#!/bin/bash
#python -u run_train.py --dropout_rate=0.3 --epoch=1000 --ngpu=1 --batch_size=256 --num_workers=0
#python -u run_train.py --dropout_rate=0.3 --epoch=1000 --ngpu=0 --batch_size=256 --num_workers=0
#python -u run_train.py --dropout_rate=0.3 --epoch=1000 --ngpu=0 --batch_size=256 --num_workers=0
#python -u run_train.py --dropout_rate=0.3 --epoch=1 --ngpu=0 --batch_size=256 --num_workers=0
#python -u run_train.py --dropout_rate=0.3 --epoch=1000 --ngpu=8 --batch_size=256 --num_workers=0
#python -u run_train.py --dropout_rate=0.3 --epoch=10 --ngpu=0 --batch_size=256 --num_workers=0

#
# 7차 테스트
#
#python -u run_train.py --dropout_rate=0.5 --epoch=1000 --ngpu=0 --batch_size=64 --num_workers=0 --lr=0.00002

#
# 8차 테스트
#  + learning rate 변경: 0.00002 --> 0.00001
#
#python -u run_train.py --dropout_rate=0.5 --epoch=1000 --ngpu=0 --batch_size=64 --num_workers=0 --lr=0.00001

#
# 9차 테스트
#  + learning rate 변경: 0.00001 --> 0.000005
#
#python -u run_train.py --dropout_rate=0.5 --epoch=1000 --ngpu=0 --batch_size=64 --num_workers=0 --lr=0.000005

#
# 10차 테스트
#  + learning rate 변경: 0.000005 --> 0.000002
#  + epoch 1000 --> 2000
#
#python -u run_train.py --dropout_rate=0.5 --epoch=2000 --ngpu=0 --batch_size=64 --num_workers=0 --lr=0.000002

#
# 11차 테스트
#  + epoch 2000 --> 3000
#
screen -L -Logfile train-out-11.txt python -u run_train.py --dropout_rate=0.5 --epoch=3000 --ngpu=0 --batch_size=64 --num_workers=0 --lr=0.000002
