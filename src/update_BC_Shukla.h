/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   update_BC_Shukla.h
 * Author: user
 *
 * Created on November 25, 2020, 3:12 PM
 */

#ifndef UPDATE_BC_SHUKLA_H
#define UPDATE_BC_SHUKLA_H

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
#include "parallel_Shukla.h"

#include <mpi.h>

using namespace std;

void add_Affected_BCC(vector<component_t>& affected_Bccs, edge_t& e, component_t& comp){
//    printf("\nComponents Sum [%d]\n", comp.sum_of_bcc);
    if(affected_Bccs.size() == 0) {
        comp.edges_affected.push_back(e);
        affected_Bccs.push_back(comp);
    } 
    else 
    {
        bool found = false;
        // Other wise need to check if the BCC has already been added to the affected list
        for(int j = 0; j < affected_Bccs.size(); j++) {
            
            if(affected_Bccs[j].sum_of_bcc == comp.sum_of_bcc)
            {
                affected_Bccs[j].edges_affected.push_back(e);
                found = true;
                break;
            } 
        }
        
        if(!found)
        {
            comp.edges_affected.push_back(e);
            affected_Bccs.push_back(comp);
        }
    }
}

/*
 * Update a graphs Betweenness Centrality values using Jamour's algorithm
 * Output the timing details as well
 */
void update_Graph_BC_Shukla(
            graph_t         graph,
            vector<edge_t> edges_vec,
            bool            compare_with_brandes = true,
            int             num_threads = 1,
            operation_t     operation = INSERTION
        )
{
    // insert all edges into this one
    graph_t graph_prime = graph;
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status    status;
    
    timer           tm;
    timer           tm_findBCC;
    timer           tm_UpdateBCC;
    double          brandes_time = 1.0;
    vector<double>  BC_vec;
    vector<double>  tm_vec;
    vector<double>  speedup_vec;
    
    BC_vec.resize(graph.size());
    if(compare_with_brandes) {
        tm.start();
//        fast_brandes_BC(graph, BC_vec);  // this is faster, uses components
//        BC_vec = brandes_bc(graph);
        tm.stop();
        brandes_time = tm.interval();
    }
    if(rank == 0) {
        printf("Graph[%s]  V[%d]  E[%d]  Brandes_time[%.6f]\n",
                graph.graph_name.c_str(),
                graph.size(),
                graph.edge_set.size(),
                brandes_time);
    }
    
    tm.start();
    tm_findBCC.start();
    
    // Array of BCCs that have been affected, these are BCC in G (but were found in G')
    vector<component_t> affected_Bccs;
    
    // add all the edges into G prime
    for(int i = 0; i < edges_vec.size(); ++i) {
        edge_t e = edges_vec[i];
        
        if(operation == INSERTION) {
            //IMP: assumes @e is not in @graph, @e will not be in @comp
            
            // biconnected component of G' that edge e belongs to
            graph_prime.insert_edge(e.first, e.second);
        }
    }
    
    printf("Number of edges to insert [%d]\n", edges_vec.size());
    printf("Find all the affected BCCs\n");
    // Find all the affected BCCs for the edges
    for(int i = 0; i < edges_vec.size(); ++i) {
        
        edge_t e = edges_vec[i];
//        printf("Edge: [%d], [%d]", e.first, e.second);
        
        // biconnected component of G' if insertion, G if deletions
        component_t comp;
        comp.comp_type = BCC;
        
        if(operation == INSERTION) {
            // find the BCC of G'
            graph_prime.find_edge_bcc_prime(comp, e, "INSERTION");
            // find the BCC of G
            
        } else if(operation == DELETION) {
            
            //IMP: assumes @e is not in @graph, @e will not be in @comp
            //     (remove @e from @graph)
            graph.remove_edge(e.first, e.second);
            graph.find_edge_bcc(comp, e, "DELETION");
            graph.insert_edge(e.first, e.second);
        }
        
        //map @e to the new edge in the comp
        e.first  = comp.subgraph.outin_label_map[e.first];
        e.second = comp.subgraph.outin_label_map[e.second];
        
        if(operation == DELETION) {
            // Add back in the edge, so this is BCC (not BCC')
            comp.subgraph.insert_edge(e.first, e.second);
        }
        
        // Add the biconnected component to the list of affected BCCs
        add_Affected_BCC(affected_Bccs, e, comp);
    }
    
    tm_findBCC.stop();
    printf("There are [%d] affected BCCs\n", affected_Bccs.size());
    printf("time[%.6f] for Finding Affected BCC \n", tm_findBCC.interval());
    printf("Find all the affected Nodes\n");
    // Find the affected nodes within each of the subgraphs
    // TODO: could parallelize this, to update each BCC on different threads
    
    timer tm_To_Find_Nodes;
    tm_To_Find_Nodes.start();
    find_Affected_Nodes(affected_Bccs);
    tm_To_Find_Nodes.stop();
    printf("Time[%.6f] to find the affected Nodes shukla\n", tm_To_Find_Nodes.interval());
    
//    for(int z = 0; z < affected_Bccs.size(); ++z) {
//        printf("Printing BCC [%d]\n", z);
//        affected_Bccs[z].print();
//    }
    
    tm_UpdateBCC.start();
    
    vector<double> dBC_vec;
    for(int i = 0; i < affected_Bccs.size(); i++)
    {
        timer tm_interval;
        tm_interval.start();
        // for each affected BCC S in G, remove the source dependencies
        // for each affected BCC S' in G', add the source dependencies
        parallel_Shukla(dBC_vec, affected_Bccs[i], num_threads, operation);
        
        tm_interval.stop();
        printf("Time to finish updating BCC: [%d], it took  [%.6f]\n", i, tm_interval.interval());
    }
    
    tm_UpdateBCC.stop();
    printf("Time to update BC in nodes [%.6f] for shukla\n", tm_UpdateBCC.interval());
    
    tm.stop();
    double e_time = tm.interval();
    double e_speedup = brandes_time/e_time;
    printf("time[%.6f]  speed-up[%.6f]\n", e_time, e_speedup);
    
//     print all the affected Bccs
    
    
    
    
    // Printing info -- uncomment eventually
    
//    double tm_mean, tm_stddev, tm_median, tm_min, tm_max;
//    simple_stats(tm_vec, tm_mean, tm_stddev, tm_median, tm_min, tm_max);
}



#endif /* UPDATE_BC_SHUKLA_H */

