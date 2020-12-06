

#include "utility.h"
#include <algorithm>
#include <numeric>
#include <math.h>
#include <iostream>

using namespace std;

void simple_stats(vector<double> vec,
        double& mean,
        double& stddev,
        double& median,
        double& min,
        double& max)
{
    sort(vec.begin(), vec.end());
    min = *(min_element(vec.begin(), vec.end()));
    max = *(max_element(vec.begin(), vec.end()));
    double sum = accumulate(vec.begin(), vec.end(), 0.0);
    mean = sum/vec.size();
    median = vec[vec.size()/2];
    
    double sum_2 = 0;
    for(int i = 0; i < vec.size(); ++i)
        sum_2 += ((vec[i]-mean)*(vec[i]-mean));
    stddev = sqrt(sum_2/vec.size());
    
}


string extract_graph_name(string path)
{
    string out;
    int pos = path.rfind("/") + 1;
    int length = path.length() - pos + 1;
    out = path.substr(pos, length);
    return out;
}

/*
 * generates @num_edges random edges that are not in @graph
 */
void gen_rand_edges(int num_edges,
        graph_t& graph,
        vector<edge_t>& out_vec)
{
    out_vec.clear();
    for(int i = 0; i < num_edges; ++i) {
        edge_t rand_edge;
        node_id_t src, dst;
        do {
            src = rand()%graph.size();
            dst = rand()%graph.size();
            rand_edge.first    = src;
            rand_edge.second   = dst;
        } while(graph.has_edge(rand_edge) ||
                find(out_vec.begin(), out_vec.end(), rand_edge) != out_vec.end());
        out_vec.push_back(rand_edge);
    }
}

/*
 * generates @num_edges random edges that are in @graph, but are not bridges
 */
void gen_rand_edges_deletions(int num_edges,
        graph_t& graph,
        vector<edge_t>& out_vec)
{
    vector<edge_t> bridges_vec;
    graph.find_bridge_edges(bridges_vec);
    //NOTE:insert each edge twice -- e(v1, v2) e(v2, v1)
    int num_bridges = bridges_vec.size();
    for(int i = 0; i < num_bridges; ++i) {
        bridges_vec.push_back(make_pair(bridges_vec[i].second, bridges_vec[i].first));
    }
    out_vec.clear();
    int num_edges_graph = graph.edge_set.size();
    set<edge_t>::const_iterator it(graph.edge_set.begin());
    for(int i = 0; i < num_edges; ++i) {
        edge_t rand_edge;
        do {
            //select an edge at random
            it = graph.edge_set.begin();
            advance(it, rand()%num_edges_graph);
            rand_edge = *it;
        } while(find(bridges_vec.begin(), bridges_vec.end(), rand_edge) != bridges_vec.end());
        out_vec.push_back(rand_edge);
    }
}