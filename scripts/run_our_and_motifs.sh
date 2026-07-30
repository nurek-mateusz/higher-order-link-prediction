#!/usr/bin/env bash
for ds in coauth-MAG-Geology coauth-MAG-History contact-high-school contact-primary-school \
          email-Eu email-Enron NDC-classes NDC-substances tags-ask-ubuntu threads-ask-ubuntu; do
    for m in xgb rf lr; do
        for k in 1 3 5 10 20 40; do
            echo "=== $ds $m $k ==="
            python train_model.py "$ds" "$m" b "$k"
        done
    done
done