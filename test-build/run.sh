
# construction phase: ./run.sh 0
# search phase: ./run.sh 1

#vecsize=17770
#vecdim=300
#dataset="netflix"

#vecsize=136736
#vecdim=300
#dataset="yahoomusic"

#vecsize=999000
#vecdim=100
#dataset="music100"

vecsize=2339373
vecdim=150
dataset="imagenet"

#vecsize=10000000
#vecdim=128
#dataset="sift10m"

topk=10
efConstruction=1024
M=32

prefix="/data/liujie/gqr/data/${dataset}"
lshboxPath="${prefix}/${dataset}_top${topk}_product_groundtruth.lshbox"
#lshboxPath="${prefix}/${dataset}_groundtruth.lshbox"
basePath="${prefix}/${dataset}_base.fvecs"
queryPath="${prefix}/${dataset}_query.fvecs"

threshold=$2
proj_dir="/data/liujie/ip-nsw/test-models"
output_graph="${proj_dir}/${dataset}/${dataset}_out_graph_${M}_${efConstruction}.hnsw.${threshold}"
log_file="${dataset}_log_${threshold}.txt"


#++++++++++++++++++++++++++++++++++
#angularlshboxPath="${prefix}/${dataset}_top100_angular_groundtruth.lshbox"
# --benchmark2 $angularlshboxPath
#++++++++++++++++++++++++++++++++++

#cmake ./ 
make  2>&1 | tee log.txt

# 0 -> construction ; 1 -> search
# gdb --args 
if [ $1 -eq 0 ]; then
    time ./main --mode database --database $basePath --databaseSize $vecsize --dimension $vecdim --outputGraph $output_graph --efConstruction $efConstruction --M $M
elif [ $1 -eq 1 ]; then
    time ./main --mode query --query $queryPath --querySize 1000 --dimension $vecdim --benchmark $lshboxPath --inputGraph $output_graph --topK $topk --efSearch 128 --output result.txt | tee $log_file
fi
