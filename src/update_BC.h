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

#include "graph_t.h"
#include "bc.h"
#include "experiments.h"
#include "utility.h"
#include "types.h"

#include <mpi.h>

/*
 * Actual timing details are done here
 */
void update_Graph_BC(
            graph_t         graph,
            vector<edge_t> edges_vec,
            bool            compare_with_brandes = true,
            int             num_threads = 1,
            bool            del_edge = false,
            operation_t     op = INSERTION
        )
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status    status;
    
    timer           tm;
    double          brandes_time = 1.0;
    vector<double>  BC_vec;
    vector<double>  tm_vec;
    vector<double>  speedup_vec;
    
    BC_vec.resize(graph.size());
    if(compare_with_brandes) {
        tm.start();
        fast_brandes_BC(graph, BC_vec);  // this is faster, uses components
//        BC_vec = brandes_bc(graph);
        tm.stop();
        brandes_time = tm.interval();
    }
    if(rank == 0) {
        printf("Graph[%s]  V[%d]  E[%d]  Brandes_tm[%.2f]\n",
                graph.graph_name.c_str(),
                graph.size(),
                graph.edge_set.size(),
                brandes_time);
    }
    for(int i = 0; i < edges_vec.size(); ++i) {
        edge_t e = edges_vec[i];
        if(del_edge) graph.remove_edge(e.first, e.second);
        tm.start();
        Update_BC(BC_vec, graph, BCC, e, num_threads, op); // Use BCC for the algorithm
        tm.stop();
        double e_time = tm.interval();
        tm_vec.push_back(e_time);
        double e_speedup = brandes_time/e_time;
        speedup_vec.push_back(e_speedup);
        
        if(rank == 0) {
            printf("e(%-6d,%-6d)  time[%.2f]  speed-up[%.2f]\n",
                    e.first,
                    e.second,
                    e_time,
                    e_speedup);
        }
        if(del_edge) graph.insert_edge(e.first, e.second);
        //synchronization barrier so no one starts next edge before others
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    double tm_mean, tm_stddev, tm_median, tm_min, tm_max;
    double su_mean, su_stddev, su_median, su_min, su_max;
    simple_stats(tm_vec, tm_mean, tm_stddev, tm_median, tm_min, tm_max);
    simple_stats(speedup_vec, su_mean, su_stddev, su_median, su_min, su_max);
    
    if(rank == 0)
        printf("Avg.tm[%.2f]  Avg.sup[%.2f]\n\n", tm_mean, su_mean);
}

#endif /* UPDATE_BC_H */

