# Understanding and Improving Proximity Graph based Maximum Inner Product Search (AAAI-20)

This is the implementation for ```ip-nsw+``` from the following paper:

Jie Liu*, Xiao Yan*, Xinyan Dai, Zhirong Li, James Cheng, Ming-Chang Yang. [Understanding and Improving Proximity Graph based Maximum Inner Product Search](https://arxiv.org/abs/1909.13459). AAAI Conference on Artificial Intelligence (AAAI), 2020.

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
Yahoo!Music,word2vec and ImageNet can be found here: http://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html

## Run experiments
```
Usage: main [OPTIONS]
This tool supports two modes: to construct the graph index from the database and to retrieve top K maximum inner product vectors using the constructed index. Each mode has its own set of parameters.

  --mode                "database" or "query". Use "database" for
                        constructing index and "query" for top K
                        maximum inner product retrieval

Database mode supports the following options:
  --database            Database filename. Database should be stored in .fvecs format.
  --databaseSize        Number of vectors in the database
  --dimension           Dimension of vectors
  --outputGraph         Filename for the output index graph
  --efConstruction      efConstruction parameter in MIPS graph. Default: 16
  --M                   M parameter. Default: 32
  --cos_efConstruction  efConstruction parameter in angular graph. Default: 100
  --cos_M               M parameter in angular graph. Default: 10

Query mode supports the following options:
  --query               Query filename. Queries should be stored in .fvecs format.
  --querySize           Number of queries
  --dimension           Dimension of vectors
  --benchmark           Groundtruth top-K results for query. 
                        You can refer to [GQR project](https://github.com/lijif2/gqr) for the details of obtaining benchmark.
                        Should use cal_groundtruth.sh in gqr/script folder
  --inputGraph          Filename for the input index graph
  --efSearch            efSearch parameter in MIPS graph. Default: 128
  --cos_efSearch        efSearch parameter in angular graph. Default: 1
  --topK                Top size for retrieval. Default: 1
  --output              Filename to print the result. Default: stdout
```
