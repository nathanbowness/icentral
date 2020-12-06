/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   update_BC_Batch.h
 * Author: user
 *
 * Created on November 25, 2020, 3:12 PM
 */

#ifndef UPDATE_BC_BATCH_H
#define UPDATE_BC_BATCH_H

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
#include "parallel_Batch_BC.h"

#include <mpi.h>

using namespace std;

void add_Affected_BCC(vector<component_t>& affected_Bccs, edge_t& e, component_t& comp){
    if(affected_Bccs.size() == 0) {
        comp.edges_affected.push_back(e);
        affected_Bccs.push_back(comp);
    } 
    else 
    {
        bool found = false;
        // Need to check if the BCC has already been added to the affected list
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
void update_Graph_BC_Batch(
            graph_t         graph,
            vector<edge_t> edges_vec,
            int             num_threads = 1,
            operation_t     operation = INSERTION
        )
{    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status    status;
    
    timer           overall_Time;
    timer           tm_findBCC;
    timer           tm_UpdateBCC;
    vector<double>  BC_vec;
    
    BC_vec.resize(graph.size());

    if(rank == 0) {
        printf("\n\nRunning Batch iCentral on Graph[%s]  V[%d]  E[%d]\n",
                graph.graph_name.c_str(),
                graph.size(),
                graph.edge_set.size());
    }
    
    overall_Time.start();
    tm_findBCC.start();
    
    // Array of BCCs that have been affected, these are BCC in G (but were found in G')
    vector<component_t> affected_Bccs;
    // For insertions, we will look for the BCCs of G'
    graph_t graph_prime = graph;
    
    if(operation == INSERTION) {    
        // add all the edges into G prime
        for(int i = 0; i < edges_vec.size(); ++i) {
            edge_t e = edges_vec[i];
            // biconnected component of G' that edge e belongs to
            graph_prime.insert_edge(e.first, e.second);
        }
        BC_vec.resize(graph_prime.size());
    }
    
    // Find all the affected BCCs for the edges
    for(int i = 0; i < edges_vec.size(); ++i) {
        
        // The biconnected component of G' if insertion, G if deletions
        component_t comp;
        comp.comp_type = BCC;
        
        edge_t e = edges_vec[i];
        
        if(operation == INSERTION) {
            // find the BCCs of G'
            graph_prime.find_edge_bcc_prime(comp, e, "INSERTION");
            
        } else if(operation == DELETION) {
            
            //IMP: assumes @e is not in @graph, @e will not be in @comp
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
    printf("It took [%.6f]s to find the affected BCCs \n", tm_findBCC.interval());

    timer tm_To_Find_Nodes;
    tm_To_Find_Nodes.start();
    find_Affected_Nodes(affected_Bccs);
    tm_To_Find_Nodes.stop();
    printf("It took [%.6f]s to find the affected Nodes\n", tm_To_Find_Nodes.interval());
    
//    for(int z = 0; z < affected_Bccs.size(); ++z) {
//        printf("Printing BCC [%d]\n", z);
//        affected_Bccs[z].print();
//    }
    
    tm_UpdateBCC.start();
    vector<double> dBC_vec;
    for(int i = 0; i < affected_Bccs.size(); i++)
    {
        // for each affected BCC S in G, remove the source dependencies
        // for each affected BCC S' in G', add the source dependencies
        parallel_Shukla(dBC_vec, affected_Bccs[i], num_threads, operation);
        
        // Synchronization barrier so no processor starts on the next BCC before others
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    for(int i = 0; i < dBC_vec.size(); ++i) {
            node_id_t actual_node_id = affected_Bccs[i].subgraph.inout_label_map[i];
            BC_vec[actual_node_id] += dBC_vec[i];
        }
    
    tm_UpdateBCC.stop();
    printf("Time to update BC in nodes [%.6f] for batch iCentral\n", tm_UpdateBCC.interval());
    
    overall_Time.stop();
    printf("The overall time to update BC in nodes [%.6f] for batch iCentral\n", overall_Time.interval());
}


#endif /* UPDATE_BC_SHUKLA_H */