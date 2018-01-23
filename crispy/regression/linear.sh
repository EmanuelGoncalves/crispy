#!/bin/bash

rm linear.c linear.o linear.cpython-36m-darwin.so

python linear_setup.py build_ext --inplace

mv crispy/regression/linear.cpython-36m-darwin.so .
mv build/temp.macosx-10.9-x86_64-3.6/linear.o .

rm -rf build/
rm -rf crispy/
