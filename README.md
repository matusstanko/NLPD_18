# NLPD_18

# Part1

Tagging_CPU.py
This file is written to work using CPU. We used this on HCP, but since CPU is not best alternative for speed, we decided to use GPU version

Tagging_GPU.py
This file is using GPU, to tag and create csv file as statement,label,label_binary,A_raw_entities,B_raw_entities,C_raw_entities
Also we added Batching, so its not tagging line by line, but works in baches of 8

main.ipynb 
Used to load created csv file


main_workflow.ipynb
Used on sample data, to get overview how NER algs. tag