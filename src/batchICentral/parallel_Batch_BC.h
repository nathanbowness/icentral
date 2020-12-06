/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   parallel_Shukla.h
 * Author: user
 *
 * Created on November 29, 2020, 4:51 PM
 */

#ifndef PARALLEL_BATCH_BC_H
#define PARALLEL_BATCH_BC_H

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <stack>
#include <queue>
#include <numeric>
#include <cmath>

#include "../utility.h"

#include <thread>
#include <mpi.h>
#include "atomic_Batch_ICentral.h"

using namespace std;


void parallel_Shukla(
        vector<double>&         dBC_vec,
        component_t&            affected_Bcc,
        int                     num_threads,
        operation_t             operation
        )
{

    vector<node_id_t> nodes_affected = affected_Bcc.nodes_affected;

    // Each machine must have only its share of the nodes_affected per BCC
    // then continue normally, note that each process will finish
    // and have it's contribution in its dBC_vec
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status    status;
    if(nodes_affected.size() < size)
        return;

    // Assign a machine a subset of the nodes_affected (set of nodes Q)
    vector<node_id_t> machine_source_vec;
    int num_s_per_machine = nodes_affected.size()/size;
    int start, end;
    start   = rank * num_s_per_machine;
    end     = start + num_s_per_machine;
    if(rank == size - 1)
        end = nodes_affected.size();
    for(int i = start; i < end; ++i) {
        machine_source_vec.push_back(nodes_affected[i]);
    }
    nodes_affected = machine_source_vec;
    //printf("RANK[%d] -- num sources [%d]\n", rank, all_sources_vec.size());

    // Assign a thread a subset of the nodes_affected (set of nodes Q)
    // This will split the array further if multiple threads per machine
    vector<vector<node_id_t> > thread_source_vec;
    thread_source_vec.resize(num_threads);
    if(nodes_affected.size() < num_threads)
        return;
    int num_s_per_thread = nodes_affected.size()/num_threads;
    int t = -1;
    for(int i = 0; i < nodes_affected.size(); ++i) {
        if(i%num_s_per_thread == 0 && t < num_threads-1)
            t++;
        thread_source_vec[t].push_back(nodes_affected[i]);
    }

    // List of threads
    vector<std::thread> thread_vec;
    thread_vec.resize(num_threads);

    // delta BC for each thread
    vector<vector<double> > dBC_vec_vec;
    dBC_vec_vec.resize(num_threads);
    
    // Start the threads
    for(int t = 0; t < num_threads; ++t) {
        thread_vec[t] = std::thread(iCentral_Batch_Block, &(dBC_vec_vec[t]), &affected_Bcc, &(thread_source_vec[t]), &operation);
    }
    
    // Wait for the threads to finish
    for(int t = 0; t < num_threads; ++t) {
        thread_vec[t].join();
    }
    
    // Accumulate the dBC_vec from each thread into the over dBC_vec
    for(int t = 0; t < num_threads; ++t) {
        for(node_id_t v = 0; v < dBC_vec.size(); ++v) {
            dBC_vec[v] += dBC_vec_vec[t][v];
        }
    }
    
    /// For MPI programming this flag must be set to true
    bool mpiProgramming = false;
    if (mpiProgramming)
    {
        //now dBC_vec of each machine is ready.
        //master gets and accumulates all dBC_vec from everyone
        printf("R[%d] done\n", rank);
        if(rank == 0) {
            //receive dBC_vec from everyone and accumulate
            timer tm;
            tm.start();
            vector<double> rcv_dBC_vec;
            rcv_dBC_vec.resize(dBC_vec.size());
            for(int p = 1; p < size; ++p) {
                MPI_Recv(&rcv_dBC_vec[0], rcv_dBC_vec.size(), MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                for(int i = 0; i < dBC_vec.size(); ++i) {
                    dBC_vec[i] += rcv_dBC_vec[i];
                }
            }
            tm.stop();
            printf("\t\tR[%d] -- Accumulation time: [%f]\n", rank, tm.interval());
        } else {
            //just send dBC_vec
            MPI_Send(&dBC_vec[0], dBC_vec.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    // Write the delta BC for each BCC to a file
    FILE* fout = fopen("dBC_whole_batch", "w");
    for(int i = 0; i < dBC_vec.size(); ++i) {
        fprintf(fout, "%f\n", dBC_vec[i]);
    }
    fclose(fout);
}


void find_Affected_Nodes(vector<component_t>& affected_Bccs){
    for(int i = 0; i < affected_Bccs.size(); ++i){
        for(int j = 0; j < affected_Bccs[i].edges_affected.size(); ++j){
            
            
            edge_t e = affected_Bccs[i].edges_affected[j];
            
            // Breadth-first search from v1, v2 of edge to compute d
            vector<int> d_src_vec, d_dst_vec;
            affected_Bccs[i].subgraph.find_sssp(e.first, d_src_vec);
            affected_Bccs[i].subgraph.find_sssp(e.second, d_dst_vec);
            
            // Add s to Q (set of nodes which the source dependencies delta_s (ds)
            // changed with the insertion of egde e)
            for(node_id_t s = 0; s < affected_Bccs[i].subgraph.size(); ++s) {
                if(d_src_vec[s] != d_dst_vec[s]) {
                    affected_Bccs[i].add_Affected_Node(s);
                }
            }
        }
    }
}

#endif /* PARALLEL_SHUKLA_H */

