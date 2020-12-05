#include <iostream>
#include <stdio.h>
#include <fstream>
#include <list>
#include <algorithm>
#include <cmath>

#include "graph_t.h"
#include "bc.h"
#include "experiments.h"
#include "utility.h"
#include "update_BC.h"
#include "update_BC_Shukla.h"
#include "unit_tests.h"

#include <stdio.h>
#include <string.h>
#include <mpi.h>

using namespace std;

vector<string> fill_path_vec()
{
    vector<string> vec;
    string path_f  = "/home/jamourft/Desktop/Research/Betweenness-Centrality/data/fj_lcc_graphs/";
    string graphs_arr[] = 
        {   
          "Erdos02.lcc.net"
        , "Erdos972.lcc.net"
        , "Cagr.lcc.net"
        , "Eva.lcc.net"
        , "Epa.lcc.net"
        , "Contact.lcc.net"
        , "Wiki-Vote.lcc.net"
        };
    
    for(int i = 0; i < 7; ++i) {
        string s = path_f + graphs_arr[i];
        vec.push_back(s);
    }
    return vec;
}

void createEdgesFromFile(vector<string>& path_to_graphs, vector<vector<edge_t>>& edge_vec2){
    for(int i = 0; i < path_to_graphs.size(); ++i) {
        vector<edge_t> edge_vec;
        string edge_file_path = path_to_graphs[i] + ".edges";
        FILE* fin = fopen(edge_file_path.c_str(), "r");
        int v1, v2;
        while(fscanf(fin, "%d %d\n", &v1, &v2) != EOF) {
            edge_vec.push_back(make_pair(v1, v2));
        }
        edge_vec2.push_back(edge_vec);
        fclose(fin);
    }
}

void printExpectedArguments(int rank){
    if(rank == 0) {
        printf("Pass one parameter, path with experiment details\n");
        printf("deletion\n");
        printf("num_edges, num_threads, rand_seed\n");
        printf("external_edges, do_icent, do_bcc_icent\n");
        printf("do_fast_brandes, do_brandes, do_qube, do_inc_qube\n");
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
    bool                    do_icent, do_bcc_icent, edges_from_file;
    bool                    do_fast_brandes, do_brandes, do_qube, do_inc_qube;
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
        int t1, t2, t3, t4;
        fscanf(fin, "%d, %d, %d;", &t1, &t2, &t3);
        edges_from_file = (t1 != 0);
        do_icent        = (t2 != 0);
        do_bcc_icent    = (t3 != 0);
        fscanf(fin, "%d, %d, %d, %d;", &t1, &t2, &t3, &t4);
        do_fast_brandes = (t1 != 0);
        do_brandes      = (t2 != 0);
        do_qube         = (t3 != 0);
        do_inc_qube     = (t4 != 0);
        
        char buff[1024*4];
        while(fscanf(fin, "%s", buff) != EOF) {
            string path = buff;
            path_to_graphs.push_back(path);
        }
        fclose(fin);

        if(edges_from_file) {
            // Edges to add/remove from the graph, specified from a file
            // returns new_edges_per_graph
            createEdgesFromFile(path_to_graphs, new_edges_per_graph);
        } else {
            // Random edges to add/remove from a graph based on a seed number
            // returns new_edges_per_graph
            createRandomEdgesFromSeed(rank, size, num_edges, rand_seed, op,
                    path_to_graphs, new_edges_per_graph, status);
        }
    }
      
    if(do_bcc_icent) {
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
//            update_Graph_BC_Jamour( // Jamour's algorithm
//                graph,
//                new_edges_per_graph[i], 
//                do_brandes, 
//                num_threads,
//                op
//                );
            update_Graph_BC_Shukla(  // Shukla's algorithm
                graph,
                new_edges_per_graph[i], 
                do_brandes, 
                num_threads,
                op
                );
            //synchronization barrier so that no one starts next graph before others
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
}

int main( int argc, char *argv[] )
{
    int i;
    int rank;
    int size;
    MPI_Status    status;
    char str_message[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    kdd_exp_main(argc, argv, rank, size);
    
    // Other methods -- need to uncomment below as well, and switch project config
//    paper_exp_main(argc, argv);
  
    MPI_Finalize();
    return 0;
}


// Additional methods for finding betweenness centrality
//void do_paper_exp(int num_edges,
//        int num_iter,
//        double max_time,
//        int rand_seed,
//        vector<string> path_vec,
//        bool do_inc_brandes,
//        bool do_qube,
//        bool do_inc_qube,
//        bool do_fuad) 
//{     
//    //to make sure the exact same edges are used for my system and QUBE
//    //I pass the same edges to both systems
//    
//    vector<double>  brandes_time_vec;
//    
//    // List of random edges 
//    vector<vector<edge_t> > edge_vec2;
//    
//    printf("Reading graphs and generating edges...\n");
//    printf("========================================\n");
//    for(int i = 0; i < path_vec.size(); ++i) {
//        graph_t graph;
//        string path = path_vec[i].c_str();
//        graph.read_graph(path);
//        graph.graph_name = extract_graph_name(path_vec[i]);
//        vector<edge_t> edge_vec;
//        gen_rand_edges(num_edges, graph, edge_vec);
//        edge_vec2.push_back(edge_vec);
//    }
//    
//    printf("\n\n\n");
//    printf("Starting Brandes...\n");
//    printf("========================================\n");
//    for(int i = 0; i < path_vec.size(); ++i) {
//        graph_t graph;
//        string path = path_vec[i].c_str();
//        graph.read_graph(path);
//        graph.graph_name = extract_graph_name(path_vec[i]);
//        double brandes_time;
//        brandes_time = exp_brandes_p(graph, num_iter, edge_vec2[i], max_time);
//        brandes_time_vec.push_back(brandes_time);
//    }
//    
//    if(do_inc_brandes) {
//        printf("\n\n\n");
//        printf("Starting Incremental-Brandes...\n");
//        printf("========================================\n");
//        for(int i = 0; i < path_vec.size(); ++i) {
//            graph_t graph;
//            string path = path_vec[i].c_str();
//            graph.read_graph(path);
//            graph.graph_name = extract_graph_name(path_vec[i]);
//            double brandes_time = brandes_time_vec[i];
//            exp_inc_brandes_p(graph, num_iter, edge_vec2[i], brandes_time);
//        }
//    }
//    
//    if(do_qube) {
//        printf("\n\n\n");
//        printf("Starting QUBE...\n");
//        printf("========================================\n");
//        for(int i = 0; i < path_vec.size(); ++i) {
//            graph_t graph;
//            string path = path_vec[i].c_str();
//            graph.read_graph(path);
//            graph.graph_name = extract_graph_name(path_vec[i]);
//            double brandes_time = brandes_time_vec[i];
//            int loc_num_iter = num_iter;
//            if(num_iter != -1 && brandes_time < max_time)
//                loc_num_iter = -1;
//            exp_qube_p(graph, loc_num_iter, edge_vec2[i], brandes_time);
//        }
//    }
//    
//    if(do_inc_qube) {
//        printf("\n\n\n");
//        printf("Starting Incremental-QUBE...\n");
//        printf("========================================\n");
//        for(int i = 0; i < path_vec.size(); ++i) {
//            graph_t graph;
//            string path = path_vec[i].c_str();
//            graph.read_graph(path);
//            graph.graph_name = extract_graph_name(path_vec[i]);
//            double brandes_time = brandes_time_vec[i];
//            int loc_num_iter = num_iter;
//            if(num_iter != -1 && brandes_time < max_time)
//                loc_num_iter = -1;
//            exp_inc_qube_p(graph, loc_num_iter, edge_vec2[i], brandes_time);
//        }
//    }
//    
//    if(do_fuad) {
//        printf("\n\n\n");
//        printf("Starting FUAD...\n");
//        printf("========================================\n");
//        for(int i = 0; i < path_vec.size(); ++i) {
//            graph_t graph;
//            string path = path_vec[i].c_str();
//            graph.read_graph(path);
//            graph.graph_name = extract_graph_name(path_vec[i]);
//            double brandes_time = brandes_time_vec[i];
//            int loc_num_iter = num_iter;
//            if(num_iter != -1 && brandes_time < max_time)
//                loc_num_iter = -1;
//            exp_fuad_p(graph, loc_num_iter, edge_vec2[i], brandes_time);
//        }
//    }
//}
//
//void paper_exp_main(int argc, char** argv)
//{
//    int             num_edges, num_iter, rand_seed;
//    bool            do_inc_brandes, do_qube, do_inc_qube, do_fuad;
//    double          max_time;
//    vector<string>  path_vec;
//    
//    if(argc != 2) {
//        printf("Pass one parameter, path with experiment details\n");
//        printf("num_edges, num_iter, max_time, rand_seed\n");
//        printf("do_inc_brandes, do_qube, do_inc_qube, do_fuad\n");
//        printf("list of graph paths\n");
//        exit(1);
//    } else {
//        FILE* fin = fopen(argv[1], "r");
//        fscanf(fin, "%d, %d, %lf, %d;\\r\\n", &num_edges, &num_iter, &max_time, &rand_seed);
//        printf("Modifying [%d edges]\n", num_edges);
//        printf("Starting with [%d iterations]\n", num_iter);
//        printf("Starting with [%d as rand_seed]\n", rand_seed);
//        int t1, t2, t3, t4;
//        fscanf(fin, "%d, %d, %d, %d;\\r\\n", &t1, &t2, &t3, &t4);
//        do_inc_brandes  = (t1 != 0);
//        do_qube         = (t2 != 0);
//        do_inc_qube     = (t3 != 0);
//        do_fuad         = (t4 != 0);
//        char buff[1024*4];
//        while(fscanf(fin, "%s\\r\\n", buff) != EOF) {
//            string path = buff;
//            path_vec.push_back(path);
//        }
//    }
//    do_paper_exp(num_edges, num_iter, max_time, rand_seed, path_vec,
//        do_inc_brandes, do_qube, do_inc_qube, do_fuad);
//}