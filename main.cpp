#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include "hnswlib.h"
#include <algorithm>
#include <ctime>
#include <omp.h>


const int defaultEfConstruction = 1024;
const int defaultEfSearch = 128;
const int defaultM = 32;
const int defaultTopK = 10;


int loadFvecs(float*& data, int numItems, std::string inputPath) {
    std::ifstream fin(inputPath, std::ios::binary);
    if (!fin) {
        std::cout << "cannot open file " << inputPath << std::endl;
        assert(false);
    }
    int dimension;
    fin.read((char*)&dimension, 4);
    data = new float[numItems* dimension];
    fin.read((char*)data, sizeof(float) * dimension);

    int dim;
    for (int i = 1; i < numItems; ++i) {
        fin.read((char*)&dim, 4);
        assert(dim == dimension);
        fin.read((char*)(data + i * dimension), sizeof(float) * dimension);
    }
    fin.close();
    return dimension;
}

std::vector<std::priority_queue<std::pair<float, labeltype >>> loadLSHBOX(std::string inputPath) {
    std::vector<std::priority_queue<std::pair<float, labeltype >>> answers;

    std::ifstream fin(inputPath.c_str());
    unsigned numQueries;
    unsigned K;
    fin >> numQueries >> K;
    answers.resize(numQueries);

    unsigned qId;
    unsigned id;
    float dist;
    int index = 0;
    for (int q = 0; q < numQueries; ++q) {
        fin >> qId;
        assert(qId == q);
        for (int i = 0; i < K; ++i) {
            fin >> id >> dist;
            answers[q].emplace(dist, id);
        }
    }
    fin.close();
    return answers;
}

void printHelpMessage()
{
    std::cerr << "Usage: main [OPTIONS]" << std::endl;
    std::cerr << "This tool supports two modes: to construct the graph index from the database and to retrieve top K maximum inner product vectors using the constructed index. Each mode has its own set of parameters." << std::endl;
    std::cerr << std::endl;

    std::cerr << "  --mode            " << "\"database\" or \"query\". Use \"database\" for" << std::endl;
    std::cerr << "                    " << "constructing index and \"query\" for top K" << std::endl;
    std::cerr << "                    " << "maximum inner product retrieval" << std::endl;
    std::cerr << std::endl;

    std::cerr << "Database mode supports the following options:" << std::endl;
    std::cerr << "  --database        " << "Database filename. Database should be stored in binary format." << std::endl;
    std::cerr << "                    " << "Vectors are written consecutive and numbers are" << std::endl;
    std::cerr << "                    " << "represented in 4-bytes floating poing format (float in C/C++)" << std::endl;
    std::cerr << "  --databaseSize    " << "Number of vectors in the database" << std::endl;
    std::cerr << "  --dimension       " << "Dimension of vectors" << std::endl;
    std::cerr << "  --outputGraph     " << "Filename for the output index graph" << std::endl;
    std::cerr << "  --efConstruction  " << "efConstruction parameter. Default: " << defaultEfConstruction << std::endl;
    std::cerr << "  --M               " << "M parameter. Default: " << defaultM << std::endl;
    std::cerr << std::endl;
    std::cerr << "Query mode supports the following options:" << std::endl;
    std::cerr << "  --query           " << "Query filename. Queries should be stored in binary format." << std::endl;
    std::cerr << "                    " << "Vectors are written consecutive and numbers are" << std::endl;
    std::cerr << "                    " << "represented in 4-bytes floating poing format (float in C/C++)" << std::endl;
    std::cerr << "  --querySize       " << "Number of queries" << std::endl;
    std::cerr << "  --dimension       " << "Dimension of vectors" << std::endl;
    std::cerr << "  --inputGraph      " << "Filename for the input index graph" << std::endl;
    std::cerr << "  --efSearch        " << "efSearch parameter. Default: " << defaultEfSearch << std::endl;
    std::cerr << "  --topK            " << "Top size for retrieval. Default: " << defaultTopK << std::endl;
    std::cerr << "  --output          " << "Filename to print the result. Default: " << "stdout" << std::endl;
    
}

void printError(std::string err)
{
    std::cerr << err << std::endl;
    std::cerr << std::endl;
    printHelpMessage();
}

int main(int argc, char** argv) {

    std::string mode;
    std::ifstream input;
    std::ifstream inputQ;
    int efConstruction = defaultEfConstruction;
    int efSearch = defaultEfSearch;
    int M = defaultM;
    int vecsize = -1;
    int qsize = -1;
    int vecdim = -1;
    std::string graphname;
    std::string outputname;
    int topK = defaultTopK;
    std::string dataname;
    std::string queryname;
    std::string benchmarkname;

    // for verify the significance of finding topk angular NNS
    std::string angularbenchmarkname;

    hnswlib::HierarchicalNSW<float> *appr_alg;
    
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--h" || std::string(argv[i]) == "--help") {
            printHelpMessage();
           return 0; 
        }
    }
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--m" || std::string(argv[i]) == "--mode") {
            if (std::string(argv[i + 1]) == "database") {
                mode = "database";
            } else if (std::string(argv[i + 1]) == "query") {
                mode = "query";
            } else {
                printError("Unknown running mode \"" + std::string(argv[i + 1]) + "\". Please use \"database\" or \"query\"");
                return 0;
            }
            break;
        }
    }
    if (mode.empty()) {
        printError("Running mode was not specified");
        return 0;
    }

    std::cout << "Mode: " << mode << std::endl;

    
    if (mode == "database") {
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--d" || std::string(argv[i]) == "--data" || std::string(argv[i]) == "--database") {
                dataname = std::string(argv[i + 1]);
                break;
            }
        }
        std::cout << "Database file: " << dataname << std::endl;



        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--dataSize" || std::string(argv[i]) == "--dSize" || std::string(argv[i]) == "--databaseSize") {
                if (sscanf(argv[i + 1], "%d", &vecsize) != 1 || vecsize <= 0) {
                    printError("Inappropriate value for database size: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (vecsize == -1) {
            printError("Database size was not specified");
            return 0;
        }
        std::cout << "Database size: " << vecsize << std::endl;
        
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--dataDim" || std::string(argv[i]) == "--dimension" || std::string(argv[i]) == "--databaseDimension") {
                if (sscanf(argv[i + 1], "%d", &vecdim) != 1 || vecdim <= 0) {
                    printError("Inappropriate value for database dimension: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (vecdim == -1) {
            printError("Database dimension was not specified");
            return 0;
        }
        std::cout << "Database dimension: " << vecdim << std::endl;
        
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--outGraph" || std::string(argv[i]) == "--outputGraph") {
                std::ofstream outGraph(argv[i + 1]);
                if (!outGraph.is_open()) {
                    printError("Cannot create file \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                outGraph.close();
                graphname = std::string(argv[i + 1]);
                break;
            }
        }
        if (graphname.empty()) {
            printError("Filename of the output graph was not specified");
            return 0;
        }
        std::cout << "Output graph: " << graphname << std::endl;


        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--efConstruction") {
                if (sscanf(argv[i + 1], "%d", &efConstruction) != 1 || efConstruction <= 0) {
                    printError("Inappropriate value for efConstruction: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "efConstruction: " << efConstruction << std::endl;
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--M") {
                if (sscanf(argv[i + 1], "%d", &M) != 1 || M <= 0) {
                    printError("Inappropriate value for M: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "M: " << M << std::endl;
    


       
        hnswlib::L2Space l2space(vecdim);
        float *mass = NULL;
        size_t datadim = loadFvecs(mass, vecsize, dataname);

        //  calculate norms for angular queue   ---- jie 2019-01-14
        std::vector<float> element_norms;
        element_norms.reserve(vecsize);
        for (int i = 0; i < vecsize; ++i) {
            float line_norm = 0;
            for (int j = 0; j < vecdim; ++j) {
                float ele = mass[i * vecdim + j];
                line_norm += ele * ele;
            }
            line_norm = sqrt(line_norm);
            element_norms.push_back(line_norm);
        }
		
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, vecsize, M, efConstruction);

        appr_alg->elementNorms = std::move(element_norms);
        std::cout << "testing " << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::cout << appr_alg->elementNorms[i] << std::endl;
        }

        std::cout << "Building index\n";
        double t1 = omp_get_wtime();
        for (int i = 0; i < 1; i++) {
            appr_alg->addPoint((void *)(mass + vecdim*i), (size_t)i);
        }
#pragma omp parallel for
        for (int i = 1; i < vecsize; i++) {
            appr_alg->addPoint((void *)(mass + vecdim*i), (size_t)i);
        }
        double t2 = omp_get_wtime();

        // test if cos graph is correct   jie 0506
        /*
        for (int i = 0; i < 10; ++i) {
          int *temp_data = (int *)(appr_alg->data_level0_memory_ + i * appr_alg->size_data_per_element_ + appr_alg->size_links_level0_ / 2);
          std::cout << "construction # links: " << *temp_data << std::endl;
        }
        */
 
        std::cout << "Index built, time=" << t2 - t1 << " s" << "\n";
        appr_alg->SaveIndex(graphname.data());
        delete appr_alg;
        delete mass;
    } else {
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--q" || std::string(argv[i]) == "--query") {
                queryname = std::string(argv[i + 1]);
                break;
            }
        }
        std::cout << "Query filename: " << queryname << std::endl;

        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--benchmark") {
                benchmarkname = std::string(argv[i + 1]);
                break;
            }
        }
        std::cout << "benchmark filename: " << benchmarkname << std::endl;

        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--benchmark2") {
                angularbenchmarkname = std::string(argv[i + 1]);
                break;
            }
        }
        std::cout << "angular benchmark filename: " << angularbenchmarkname << std::endl;

        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--querySize" || std::string(argv[i]) == "--qSize") {
                if (sscanf(argv[i + 1], "%d", &qsize) != 1 || qsize <= 0) {
                    printError("Inappropriate value for query size: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (qsize == -1) {
            printError("Query size was not specified");
            return 0;
        }
        std::cout << "Query size: " << qsize << std::endl;
        
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--queryDim" || std::string(argv[i]) == "--dimension" || std::string(argv[i]) == "--queryDimension") {
                if (sscanf(argv[i + 1], "%d", &vecdim) != 1 || vecdim <= 0) {
                    printError("Inappropriate value for query dimension: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (vecdim == -1) {
            printError("Query dimension was not specified");
            return 0;
        }
        std::cout << "Query dimension: " << vecdim << std::endl;
        
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--inGraph" || std::string(argv[i]) == "--inputGraph") {
                std::ifstream inGraph(argv[i + 1]);
                if (!inGraph.is_open()) {
                    printError("Cannot open file \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                inGraph.close();
                graphname = std::string(argv[i + 1]);
                break;
            }
        }
        if (graphname.empty()) {
            printError("Filename of the input graph was not specified");
            return 0;
        }
        std::cout << "Input graph: " << graphname << std::endl;


        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--efSearch") {
                if (sscanf(argv[i + 1], "%d", &efSearch) != 1 || efSearch <= 0) {
                    printError("Inappropriate value for efSearch: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "efSearch: " << efSearch << std::endl;
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--topK") {
                if (sscanf(argv[i + 1], "%d", &topK) != 1 || topK <= 0) {
                    printError("Inappropriate value for top size: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "Top size: " << topK << std::endl;
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--output" || std::string(argv[i]) == "--out") {
                std::ofstream output(argv[i + 1]);
                if (!output.is_open()) {
                    printError("Cannot open file \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                output.close();
                outputname = std::string(argv[i + 1]);
                break;
            }
        }
        if (outputname.empty()) {
            std::cout << "Output file: " << "stdout" << std::endl;
        } else {
            std::cout << "Output file: " << outputname << std::endl;
        }


        std::vector<std::priority_queue<std::pair<float, labeltype >>> answers = loadLSHBOX(benchmarkname);


        hnswlib::L2Space l2space(vecdim);
        float *massQ = NULL;
        size_t datadim = loadFvecs(massQ, qsize, queryname);

        std::priority_queue< std::pair< float, labeltype >> gt[qsize];

        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, graphname.data(), false);

        appr_alg->size_links_level0_ = appr_alg->offsetData_;
        std::cout << "422 line error found: " << appr_alg->size_links_level0_ << std::endl;

        /*
        for (int i = 100000; i < 100010; ++i) {
          int *temp_data = (int *)(appr_alg->data_level0_memory_ + i * appr_alg->size_data_per_element_ + appr_alg->size_links_level0_ / 2);
          std::cout << "number of links : " << *temp_data << std::endl;
        }
        */
        
        // construct a map between external_id and internal_id
        /*
        std::vector<int> external_internal_map(appr_alg->maxelements_);
        for (int i = 0; i < appr_alg->maxelements_; ++i) {
            external_internal_map[appr_alg->getExternalLabel(i)] = i;
        }
        */

        /*
        std::vector<std::priority_queue<std::pair<float, labeltype >>> angular_answers = loadLSHBOX(angularbenchmarkname);

        std::unordered_set<int> temp_set;
        int *temp_data = NULL;

        for (int i = 0; i < qsize; ++i) {
            std::priority_queue<std::pair<float, labeltype >> angular_gt(angular_answers[i]);
            while (angular_gt.size()) {
                int neighbor_id = angular_gt.top().second;
                temp_set.insert(neighbor_id);
                temp_data = (int *)(appr_alg->data_level0_memory_ + external_internal_map[neighbor_id] * appr_alg->size_data_per_element_);
                int size = *temp_data;
                for (int j = 1; j <= size; j++) {
                  neighbor_id = *(temp_data + j);
                  temp_set.insert(appr_alg->getExternalLabel(neighbor_id));
                }
                angular_gt.pop();
            }
            std::vector<int> temp_vec(temp_set.size());
            std::copy(temp_set.begin(), temp_set.end(), temp_vec.begin());
            appr_alg->candidates_by_cos_topk.push_back(std::move(temp_vec));
            temp_set.clear();
        }
        */
        // ===================================================================

        // =========== very important here ========================================================================================================
        /*
        std::vector<int> external_count(appr_alg->maxelements_);
        int *temp_data = NULL;
        int degree_count = 0;
        for (int i = 0; i < appr_alg->maxelements_; ++i) {
            temp_data = (int *)(appr_alg->data_level0_memory_ + i * appr_alg->size_data_per_element_);
            int degree = *temp_data;
            for (int j = 1; j <= degree; ++j) {
                external_count[appr_alg->getExternalLabel(*(temp_data + j))]++;
            }
            degree_count += degree;
            // std::cout << "norm : " << norm << ", degrees : " << degree  << std::endl;
            // std::cout << norm << ", " << degree  << std::endl;
        }
        std::cout << "avg. degree = " << (float)degree_count / appr_alg->maxelements_  << std::endl;
        for (int i = 0; i < appr_alg->maxelements_; ++i) {
            float norm = appr_alg->elementNorms[appr_alg->getExternalLabel(i)];
            std::cout << norm << ", " << external_count[appr_alg->getExternalLabel(i)] << std::endl;
        }
        */
        // ========================================================================================================================================


        std::cout << "max level : " << appr_alg->maxlevel_ << std::endl;
        std::vector<int> efs;

        // efs.push_back(100);
        for (int i = 10; i < 100; i+=10) {
            efs.push_back(i);
        }
        for (int i = 100; i < 300; i += 20) {
            efs.push_back(i);
        }
        for (int i = 300; i < 1000; i += 100) {
            efs.push_back(i);
        }
        for (int i = 2000; i < 20000; i += 2000) {
            efs.push_back(i);
        }
        //for (int i = 20000; i < 100000; i += 20000) {
        //    efs.push_back(i);
        //}

        for (int efSearch : efs) {
            appr_alg->setEf(efSearch);
            appr_alg->dist_calc = 0;
            std::ofstream fres;
            if (!outputname.empty()) {
                fres.open(outputname);
            }

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < qsize; i++) {
                gt[i] = appr_alg->searchKnn(massQ + vecdim*i, topK);
                // std::cout << "test 0505 : " << gt[i].size() << std::endl;
            }
            auto end = std::chrono::high_resolution_clock::now();
            // std::cout << "quality : " << appr_alg->quality_of_first_bot_bucket / qsize << std::endl;

            int correct = 0, total = 0;
            // jie 2019-04-15
            // int test_correct = 0, test_avg = 0;

            for (int i = 0; i < qsize; i++) {
                std::vector <int> res;
                while (!gt[i].empty()) {
                    res.push_back(gt[i].top().second);
                    gt[i].pop();
                }
                std::reverse(res.begin(), res.end());

                std::unordered_set<labeltype> g;
                std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
                total += gt.size();
                while (gt.size()) {
                    g.insert(gt.top().second);
                    gt.pop();
                }
                for (auto it : res) {
                    // to be done soon
                    if (g.find(it) != g.end())
                        correct++;

                    if (!outputname.empty()) {
                        fres << it << ' ';
                    } else {
                        std::cout << it << ' ';
                    }
                }
                
                // jie ===============================================
                /*
                test_avg += (appr_alg->candidates_by_cos_topk)[i].size();
                for (auto it : appr_alg->candidates_by_cos_topk[i]) {
                    if (g.find(it) != g.end())
                        test_correct++;
                }
                */
                // ====================================================
                
                if (!outputname.empty()) {
                    fres << std::endl;
                } else {
                    std::cout << std::endl;
                }
            }
            if (!outputname.empty()) {
                fres.close();
            }
            std::cout << "ef : " << efSearch << ", ";
            std::cout << 1.0f * correct / total << ", ";
            std::cout << "Average query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)qsize << "ms, ";
            std::cout << "dist_computations: " << appr_alg->dist_calc / (double)qsize << std::endl;
            // jie 2019-04-15 
            /*
            std::cout << "avg # of candidates : " << test_avg / (double)qsize << ", ";
            std::cout << 1.0f * test_correct / total << std::endl;
            std::cout << 1.0f * total / qsize << std::endl;
            */
        }
        delete appr_alg;
        delete massQ;
    }
    return 0;
}
