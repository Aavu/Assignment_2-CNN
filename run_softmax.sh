#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model mymodel \
    --epochs 50 \
    --weight-decay 0.0 \
    --momentum 0.95 \
    --batch-size 64 \
    --lr 0.001 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################