# R5T

R5T replaces the failed 3-channel W1A8 stem with FracBNN-style thermometer encoding.

Pipeline:

raw RGB -> thermometer R=8 -> 96 bipolar binary channels -> W1A1 3x3 stem -> existing R4 backbone

This pack preserves the R4 backbone/head and changes only the input representation and stem.

The included R4->thermometer weight projection is an engineering warm-start strategy for this project; it is not claimed as a method from the FracBNN paper.
