# Non-metric Similarity Graphs for Maximum Inner Product Search

This is the implementation for ```ip-nsw+``` from the following paper:

Jie Liu*, Xiao Yan*, Xinyan Dai, Zhirong Li, James Cheng, Ming-Chang Yang. AAAI Conference on Artificial Intelligence (AAAI), 2020.

Acknowledgement: This work improves graph-based MIPS method based on the following work: S. Morozov, A. Babenko. Non-metric Similarity Graphs for Maximum Inner Product Search. Advances in Neural Information Processing Systems 32 (NIPS 2018).

And this repo is modified from [https://github.com/stanis-morozov/ip-nsw](https://github.com/stanis-morozov/ip-nsw).

## Requirements
  - g++ with c++11 support
  - OpenMP installed

## Compilation
To download and compile the code type:
```
$ git clone https://github.com/Jerry-liujie/ip-nsw.git
$ cd ip-nsw
$ git checkout --track origin/GraphMIPS
$ mkdir build && cd build
$ cmake ..
$ make
```
## Dataset
netflix, yahoomusic and imagenet can be found here: http://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html
We will add word2vec dataset asap.


## Run experiments
```
Usage: main [OPTIONS]
This tool supports two modes: to construct the graph index from the database and to retrieve top K maximum inner product vectors using the constructed index. Each mode has its own set of parameters.

  --mode            "database" or "query". Use "database" for
                    constructing index and "query" for top K
                    maximum inner product retrieval

Database mode supports the following options:
  --database        Database filename. Database should be stored in binary format.
                    Vectors are written consecutive and numbers are
                    represented in 4-bytes floating poing format (float in C/C++)
  --databaseSize    Number of vectors in the database
  --dimension       Dimension of vectors
  --outputGraph     Filename for the output index graph
  --efConstruction  efConstruction parameter. Default: 1024
  --M               M parameter. Default: 32

Query mode supports the following options:
  --query           Query filename. Queries should be stored in binary format.
                    Vectors are written consecutive and numbers are
                    represented in 4-bytes floating poing format (float in C/C++)
  --querySize       Number of queries
  --dimension       Dimension of vectors
  --inputGraph      Filename for the input index graph
  --efSearch        efSearch parameter. Default: 128
  --topK            Top size for retrieval. Default: 1
  --output          Filename to print the result. Default: stdout
```

