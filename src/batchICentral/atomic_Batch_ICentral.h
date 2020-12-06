/* 
 * File:   atomic_ICentral_Batch.h
 * Author: Nathan Bowness
 *
 * Created on November 29, 2020, 8:32 PM
 */

#ifndef ATOMIC_BATCH_ICENTRAL_H
#define ATOMIC_BATCH_ICENTRAL_H

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <stack>
#include <queue>
#include <numeric>
#include <cmath>

#include "../bc.h"
#include "../utility.h"

#include <thread>
#include <mpi.h>

using namespace std;

bool edge_Should_Be_Deleted(component_t& comp, node_id_t one, node_id_t two) {
    
    for(int i = 0; i < comp.edges_affected.size(); i++){
        edge_t e = comp.edges_affected[i];
        
        if(one == e.first && two == e.second || one == e.second && two == e.first)
        {
            return true;
        }
    }
    
    return false;
}

void batch_Partial_BBFS_Addition(
        iter_info_t&    iter_info,  // iteration info to be computed
        component_t&    comp,       // component
        node_id_t       s          // source of the iteration
        )
{
    subgraph_t& g = comp.subgraph;
    
    // Graph all the variables calculated from the BFS
    // ?? thread safe? 
    vector<vector<node_id_t> >&  P = iter_info.P; 
    vector<int>&                 path_cnt_vec = iter_info.sigma_vec;
    vector<int>&                 path_cnt_inc_map = iter_info.sigma_inc_vec;
    vector<int>&                 dist_map = iter_info.dist_vec;
    vector<bool>&               visited_vec = iter_info.visited_vec;
    vector<node_id_t>&          S = iter_info.S;
    
    queue<node_id_t>             Q;
    
    // Add each of the new edges from the batch into the Brandes data model
    // Thread safe since each thread has different copy of the BFS data
    for(int i = 0; i < comp.edges_affected.size(); i++)
    {
        edge_t e = comp.edges_affected[i];
        
        node_id_t src, dst;
        if(dist_map[e.first] > dist_map[e.second]) {
            src = e.second;
            dst = e.first;
        } else {
            src = e.first;
            dst = e.second;
        }

        // Compute new path counts and paths
        if(dist_map[dst] != dist_map[src]+1) {
            P[dst].clear();
            P[dst].push_back(src);
            path_cnt_vec[dst] = path_cnt_vec[src];
        } else {
            P[dst].push_back(src);
            path_cnt_vec[dst] += path_cnt_vec[src];
        }

        Q.push(dst);
        dist_map[dst] = dist_map[src] + 1;
        path_cnt_inc_map[dst] = path_cnt_vec[src];
        visited_vec[dst] = true;
    }

    while (!Q.empty()) {
        node_id_t v = Q.front();
        Q.pop();
        vector<node_id_t> nbr_vec = g.get_nbrs(v);
        for (int nbr_idx = 0; nbr_idx < nbr_vec.size(); ++nbr_idx) {
            node_id_t w = nbr_vec[nbr_idx];
            if (dist_map[w] > (dist_map[v] + 1)) {
                dist_map[w] = dist_map[v] + 1;
                P[w].clear();
                P[w].push_back(v);
                path_cnt_vec[w] = 0;
                path_cnt_inc_map[w] = path_cnt_inc_map[v];
                path_cnt_vec[w] += path_cnt_inc_map[w];
                if (!visited_vec[w]) {
                    visited_vec[w] = true;
                    Q.push(w);
                }
            } else if (dist_map[w] == (dist_map[v] + 1)) {
                path_cnt_inc_map[w] += path_cnt_inc_map[v];
                path_cnt_vec[w] += path_cnt_inc_map[v];
                if (find(P[w].begin(), P[w].end(), v) == P[w].end()) {
                    P[w].push_back(v);
                }
                if (!visited_vec[w]) {
                    visited_vec[w] = true;
                    Q.push(w);
                }
            }
        }
    }


    // Fix order of S, this could be more effcient
    for (int i = 1; i < S.size(); ++i) {
        if (dist_map[S[i - 1]] > dist_map[S[i]]) {
            int j = i;
            while (dist_map[S[j - 1]] > dist_map[S[j]]) {
                node_id_t tmp = S[j - 1];
                S[j - 1] = S[j];
                S[j] = tmp;
                --j;
            }
        }
    }
}

void batch_Partial_BBFS_Deletion(
        iter_info_t&    iter_info,  // iteration info to be computed
        component_t&    comp,       // component
        node_id_t       s          // source of the iteration
        )
{
    subgraph_t& g = comp.subgraph;
    
    vector<vector<node_id_t> >&  P = iter_info.P; 
    vector<int>&                 sigma_vec = iter_info.sigma_vec;
    vector<int>&                 dist_vec = iter_info.dist_vec;
    vector<node_id_t>&           S = iter_info.S;
    
    queue<node_id_t>             Q;
    
    //for now everything is computed from scratch, so old values are irrelevant
    iter_info.init_all(g.size());
    // Assumes iter_info is initialized
    
    // IMP: careful graph_t is not thread safe!
    //g.remove_edge(e.first, e.second);
    sigma_vec[s] = 1;
    dist_vec[s] = 0;
    Q.push(s);
    while(!Q.empty()) {
        node_id_t v_i = Q.front(); Q.pop();
        S.push_back(v_i);
        for(int i = 0; i < g.nodes_vec[v_i].size(); ++i) {
            node_id_t v_n = g.nodes_vec[v_i][i];
            // For thread safety reasons, rather than delete the edge from the component
            // don't include it's source dependencies in the calculation if not needed
            if(edge_Should_Be_Deleted(comp, v_i, v_n))
             continue;
            if(dist_vec[v_n] < 0) {
                Q.push(v_n);
                dist_vec[v_n] = dist_vec[v_i] + 1;
            }
            if(dist_vec[v_n] == dist_vec[v_i] + 1) {
                sigma_vec[v_n] = sigma_vec[v_n] + sigma_vec[v_i];
                P[v_n].push_back(v_i);
            }
        }
    }
    //g.insert_edge(e.first, e.second);
}

void batch_iCentral_Iter(
        vector<double>& dBC_vec,    // delta BC of vertices
        component_t    comp,        // BCC or BCC' if insertion
        node_id_t       s,          // source of the iteration
        iter_info_t&    iter_info,  // 
        operation_t     operation
        )
{
    
    if(operation == INSERTION) {
        
        // This is BCC' remove the new edges to find proper source dependencies for G by using BBFS, RBFS
        for(int i = 0 ; i < comp.edges_affected.size(); i++)
        {
//            printf("Atomic Edge: [%d], [%d]\n", comp.edges_affected[i].first, comp.edges_affected[i].second);
            comp.subgraph.remove_edge(comp.edges_affected[i].first, comp.edges_affected[i].second);
        }
        iter_info.init_all(comp.subgraph.size());
                      
        // Brandes Breadth-first search and Brandes Reverse Breadth-first Search
        BBFS(iter_info, comp, s);
        RBFS(dBC_vec, comp, s, iter_info, false, true);
        
        batch_Partial_BBFS_Addition(iter_info, comp, s);
        RBFS(dBC_vec, comp, s, iter_info, true, false);
    } else if(operation == DELETION) {
        
        iter_info.init_all(comp.subgraph.size());
        // Brandes Breadth-first search and Brandes Reverse Breadth-first Search
        BBFS(iter_info, comp, s);
        RBFS(dBC_vec, comp, s, iter_info, false, true);
        batch_Partial_BBFS_Deletion(iter_info, comp, s);
        RBFS(dBC_vec, comp, s, iter_info, true, false);
    }

}

void iCentral_Batch_Block(
        vector<double>*     dBC_vec,
        component_t*        comp,
        vector<node_id_t>*  source_vec,
        operation_t*        op
        )
{
    fill_vec<double>(*dBC_vec, comp->subgraph.size(), 0.0);
    iter_info_t iter_info;
    for(int i = 0; i < source_vec->size(); ++i) {
        node_id_t s = (*source_vec)[i];
        batch_iCentral_Iter(*dBC_vec, *comp, s, iter_info, *op);
    }
}


#endif /* ATOMIC_ICENTRAL_BATCH_H */

