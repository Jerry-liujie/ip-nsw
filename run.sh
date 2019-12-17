# construction phase: ./run.sh 0
# search phase: ./run.sh 1

vecsize=53387
vecdim=192
dataset="audio"
qsize=200

#vecsize=136736
#vecdim=300
#dataset="yahoomusic"
#qsize=1000

#vecsize=1000000
#vecdim=100
#dataset="music100"
#qsize=1000

#vecsize=2339373
#vecdim=150
#dataset="imagenet"
#qsize=1000

#vecsize=5000000
#vecdim=384
#dataset="new_tiny5m"
#qsize=1000

#vecsize=1000000
#vecdim=300
#dataset="word2vec"
#qsize=1000

topk=10
efConstruction=$2
M=32

cos_efConstruction=100
cos_M=10

prefix="../data/${dataset}"
lshboxPath="${prefix}/${dataset}_top${topk}_product_groundtruth.lshbox"
basePath="${prefix}/${dataset}_base.fvecs"
queryPath="${prefix}/${dataset}_query.fvecs"

output_graph="${dataset}_nips_out_graph_${M}_${efConstruction}_${cos_M}_${cos_efConstruction}.hnsw"
log_file="${dataset}_log.txt"

#cmake ./
cd build
make  2>&1 | tee log.txt

# efSearch 128
# 0 -> construction ; 1 -> search
if [ $1 -eq 0 ]; then
    time ./main --mode database --database $basePath --databaseSize $vecsize --dimension $vecdim --outputGraph $output_graph --efConstruction $efConstruction --M $M --cos_efConstruction $cos_efConstruction --cos_M $cos_M
elif [ $1 -eq 1 ]; then
    time ./main --mode query --query $queryPath --querySize $qsize --dimension $vecdim --benchmark $lshboxPath --inputGraph $output_graph --topK $topk --efSearch 128 --cos_efSearch 1 --output result.txt | tee $log_file
fi
