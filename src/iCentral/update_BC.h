/* 
 * File:   update_BC.h
 * Author: Nathan Bowness
 *
 * Created on November 21, 2020, 6:19 PM
 */

#ifndef UPDATE_BC_H
#define UPDATE_BC_H

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <list>
#include <algorithm>
#include <cmath>

#include "../graph_t.h"
#include "../bc.h"
#include "../utility.h"
#include "../types.h"

#include <mpi.h>

/*
 * Update a graphs Betweenness Centrality values using Jamour's algorithm
 * Output the timing details as well
 */
void update_Graph_BC(
            graph_t         graph,
            vector<edge_t> edges_vec,
            bool            compare_with_brandes = true,
            int             num_threads = 1,
            operation_t     operation = INSERTION
        )
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status    status;
    
    timer           tm;
    timer           total_time;
    double          brandes_time = 1.0;
    vector<double>  BC_vec;
    vector<double>  tm_vec;
    vector<double>  speedup_vec;
    
    BC_vec.resize(graph.size());
    if(compare_with_brandes) {
        tm.start();
        fast_brandes_BC(graph, BC_vec);
        tm.stop();
        brandes_time = tm.interval();
    }
    if(rank == 0) {
        printf("Graph[%s]  V[%d]  E[%d]  Brandes_time[%.2f]\n",
                graph.graph_name.c_str(),
                graph.size(),
                graph.edge_set.size(),
                brandes_time);
    }
    total_time.start();
    for(int i = 0; i < edges_vec.size(); ++i) {
        edge_t e = edges_vec[i];
        tm.start();
        Update_BC(BC_vec, graph, BCC, e, num_threads, operation); // Use BCC for the algorithm
        tm.stop();
        double e_time = tm.interval();
        tm_vec.push_back(e_time);
        double e_speedup = brandes_time/e_time;
        speedup_vec.push_back(e_speedup);

        // Synchronization barrier so no one starts next edge before others
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    total_time.stop();
    double tm_mean, tm_stddev, tm_median, tm_min, tm_max;
    double su_mean, su_stddev, su_median, su_min, su_max;
    simple_stats(tm_vec, tm_mean, tm_stddev, tm_median, tm_min, tm_max);
    simple_stats(speedup_vec, su_mean, su_stddev, su_median, su_min, su_max);
    
    if(rank == 0)
        printf("Avg.time[%.2f]  Avg.speed-up[%.2f]\n\n", tm_mean, su_mean);
    printf("The Jamour total time for [%d] edges, was [%.6f]\n\n", edges_vec.size(), total_time.interval());
}

#endif /* UPDATE_BC_H */

