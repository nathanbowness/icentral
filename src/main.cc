#include <iostream>
#include <stdio.h>
#include <fstream>
#include <list>
#include <algorithm>
#include <cmath>

#include "graph_t.h"
#include "bc.h"
#include "utility.h"

#include "iCentral/update_BC.h"
#include "batchICentral/update_BC_Batch.h"

#include <stdio.h>
#include <string.h>
#include <mpi.h>

using namespace std;

void printExpectedArguments(int rank){
    if(rank == 0) {
        printf("Pass one parameter, path with experiment details\n");
        printf("deletion\n");
        printf("num_edges, num_threads, rand_seed\n");
        printf("list of graph paths\n");
        printf("if external_edges is nonzero a file with graph_name.edges is expected\n");
        printf("external edges are assumed to be in the graph\n");
        printf("external edges should not be bridges\n");
        printf("if deletion is 1, bcc_icent with deletions will be invoked");
    }
}

void createRandomEdgesFromSeed(int rank, int size, int num_edges, int rand_seed, operation_t& op, vector<string>& path_to_graphs, vector<vector<edge_t> >& edge_vec2, MPI_Status& status){
    for(int p = 0; p < path_to_graphs.size(); ++p) {
        string graph_path = path_to_graphs[p];
        graph_t graph;
        graph.read_graph(graph_path);
        graph.graph_name = extract_graph_name(graph_path);
        vector<edge_t> edge_vec;
        
        if(rank == 0) {
            srand(rand_seed);
            // Master generates random edges and sends to everyone
            if(op == INSERTION) {
                gen_rand_edges(num_edges, graph, edge_vec);
            } else if(op == DELETION) {
                gen_rand_edges_deletions(num_edges, graph, edge_vec);
            }
            
            for(int p = 1; p < size; ++p) {
                MPI_Send(&edge_vec[0], edge_vec.size()*sizeof(edge_t), MPI_CHAR, p, 0, MPI_COMM_WORLD);
            }
        } else {
            // Slaves will get random edges from the master
            edge_vec.resize(num_edges);
            MPI_Recv(&edge_vec[0], edge_vec.size()*sizeof(edge_t), MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
        edge_vec2.push_back(edge_vec);
    }
}

void kdd_exp_main(int argc, char** argv, int rank, int size)
{
    int                     num_edges, num_threads, rand_seed;
    operation_t             op;
    vector<string>          path_to_graphs;
    vector<vector<edge_t> > new_edges_per_graph;
    vector<double>          brandes_tm_vec;
    
    MPI_Status    status;
    
    if(argc != 2) {
        printExpectedArguments(rank);
        return;
    } else {
        FILE* fin = fopen(argv[1], "r");
        int del_int;
        fscanf(fin, "%d;", &del_int);
        if(del_int == 1) {
            op = DELETION; // Should delete specific edges
        } else {
            op = INSERTION; // Should insert specific edges
        }
        fscanf(fin, "%d, %d, %d;", &num_edges, &num_threads, &rand_seed);
        printf("Modifying [%d edges]\n", num_edges);
        printf("Starting with [%d threads]\n", num_threads);
        printf("Starting with [%d as rand_seed]\n", rand_seed);
        
        char buff[1024*4];
        while(fscanf(fin, "%s", buff) != EOF) {
            string path = buff;
            path_to_graphs.push_back(path);
        }
        fclose(fin);

        // Random edges to add/remove from a graph based on a seed number
        // returns new_edges_per_graph
        createRandomEdgesFromSeed(rank, size, num_edges, rand_seed, op,
                path_to_graphs, new_edges_per_graph, status);
    }
      

    if(rank == 0) {
        printf("\n\n\n");
        printf("Starting BCC+iCentral [%d threads] [%s]...\n", num_threads, (op==DELETION?"DELETION":"INSERTION"));
        printf("========================================\n");
    }
    for(int i = 0; i < path_to_graphs.size(); ++i) {
        graph_t graph;
        string path = path_to_graphs[i].c_str();
        graph.read_graph(path);
        graph.graph_name = extract_graph_name(path_to_graphs[i]);
        update_Graph_BC( // Jamour's algorithm
            graph,
            new_edges_per_graph[i], 
            true, 
            num_threads,
            op
            );
        // Synchronization barrier so that no one starts the batch algorithm before others
        MPI_Barrier(MPI_COMM_WORLD);

        update_Graph_BC_Batch(  // Algorithm based on Shukla's implementation
            graph,
            new_edges_per_graph[i], 
            num_threads,
            op
            );
        // Synchronization barrier so that no one starts next graph before others
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main( int argc, char *argv[] )
{
    int rank;
    int size;
    MPI_Status    status;
    char str_message[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    kdd_exp_main(argc, argv, rank, size);
  
    MPI_Finalize();
    return 0;
}