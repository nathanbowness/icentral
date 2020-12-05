/* 
 * File:   component_t.h
 * Author: fuad
 *
 * Created on November 24, 2014, 4:08 PM
 */

#ifndef COMPONENT_T_H
#define	COMPONENT_T_H

#include "graph_t.h"

enum comp_type_t {BCC, MUC, GRAPH};

/*
 * The component (subgraph along with other needed information)
 * that bc computation blocks operate on.
 * Could be a BCC, an MUC, or just a graph.
 */
struct component_t {
    //maps articulation points to sizes of subgraphs connected to the bcc
    //through them
    typedef tr1::unordered_map<node_id_t, vector<int> >   art_pt_map_t;
    
    subgraph_t      subgraph;
    art_pt_map_t    art_pt_map;
    comp_type_t     comp_type;
    
    vector<edge_t>  edges_affected;
    vector<node_id_t> nodes_affected;
    long sum_of_bcc = 0;
    
    void add_Affected_Node(node_id_t node) {
        bool already_Added = false;
        for(int i = 0; i < nodes_affected.size(); ++i) {
            if(nodes_affected[i] == node)
            {
                already_Added = true;
                break;
            }
        }
        
        if (!already_Added)
            nodes_affected.push_back(node);
        
    }
    
    void print()
    {
        printf("\n");
        subgraph.print_graph();
        art_pt_map_t::iterator it = art_pt_map.begin();
        for(; it != art_pt_map.end(); ++it) {
            printf("Articulation point [%d]\n", it->first);
            for(int i = 0; i < it->second.size(); ++i) {
                printf("\t%d", it->second[i]);
            }
            printf("\n");
        }
        for(int i =0; i < edges_affected.size(); i++) {
            printf("Edges that will be removed/inserted [%d],[%d]\n", 
                    edges_affected[i].first, edges_affected[i].second);
        }
        for(int i =0; i < nodes_affected.size(); i++) {
//            printf("Nodes that  will be affected [%d]\n", 
//                    nodes_affected[i]);
        }
    }
};

#endif	/* COMPONENT_T_H */

