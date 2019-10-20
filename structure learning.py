import sys

import collections as c, numpy as np, scipy.stats as sct, pandas as pd, networkx as nx
from matplotlib import pyplot as plt
import copy, time, math


def gammalog(x):
    return sum(np.log(range(1,int(x))))

def make_graph(parents):
    graph = np.matrix([[0,0]])
    for i in parents.keys():
        for j in parents[i]:
            if len(parents[i]) != 0:
                graph = np.append(graph, [[j, i]], axis=0)
    graph = np.delete(graph, 0, axis=0)
    return graph

def cycle_check(parents, ik_map):
    
    g = make_graph(parents)
    test = nx.DiGraph()
    test.add_nodes_from([i for i in range(len(ik_map))])    
    
    parents = [ik_map[i[0]][0] for i in g.tolist()]
    children = [ik_map[i[1]][0] for i in g.tolist()]
    
    for i in range(len(parents)):
        test.add_edge(parents[i], children[i])
    
    try:
        cycle = list(nx.find_cycle(test, orientation='original'))
    except:
        cycle = []
        
    return cycle

def b_score(data, parents, values):
    
    m, m_totes, alpha_totes = c.defaultdict(int), c.defaultdict(int), c.defaultdict(int)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            k = data[i,j]
            if len(parents[j]) == 0:
                parent_k = -1
            else:
                parent_k = tuple([data[i,z] for z in range(data.shape[1]) if z in parents[j]])
            m[j, parent_k, k] += 1
            m_totes[j, parent_k] += 1
    
    for key in m_totes.keys():
        alpha_totes[key] = len(values[key[0]])
    
    a = np.sum([gammalog(m[i]+1) for i in m.keys()])
    b = np.sum([-1*gammalog(alpha_totes[i] + m_totes[i]) \
                + gammalog(alpha_totes[i]) for i in m_totes.keys()])
    return a + b

def K2(data, parents, values, ik_map):
    
    m, n = data.shape
    res = b_score(data, parents, values)
    random1 = list(np.random.choice(n, n, replace = False))
    
    for i in range(n):
        count = 0
        child = random1[i]
        random2 = list(np.random.choice(n, n, replace = False))
        for j in range(n):
            parent = random2[j]
            if (parent not in parents[child] and child not in parents[parent] and parent != child and count <= 6):
                parents[child] += [parent]
                cycle = cycle_check(parents, ik_map)
                if len(cycle) == 0:
                    test = b_score(data, parents, values)
                    if test > res:
                        res = test
                        count += 1
                    else:
                        parents[child].pop()
                else:
                    parents[child].pop()

    return parents, res

def write_gph(dataset, parents, ik_map):
    file = open(dataset + ".gph", "w")
    
    g = make_graph(parents)
    parents = [ik_map[i[0]][1] for i in g.tolist()]
    children = [ik_map[i[1]][1] for i in g.tolist()]
    for i in range(len(parents)):
        edge = parents[i] + "," + children[i] +"\n"
        file.write(edge)

    file.close()

def grapherator(dataset, parents, ik_map):
    
    g = make_graph(parents)    
    res = nx.DiGraph()
    res.add_nodes_from([ik_map[i][1] for i in range(len(ik_map))])
    
    parents = [ik_map[i[0]][1] for i in g.tolist()]
    children = [ik_map[i[1]][1] for i in g.tolist()]

    for i in range(len(parents)):
        res.add_edge(parents[i], children[i])
    
    nx.draw(res, with_labels=True, node_color='lightgrey',node_size=1000, font_weight='bold')
    plt.savefig(dataset + '.png')
    plt.show()
    
    try:
        print(list(nx.find_cycle(test, orientation='original')))
    except:
        print('No cycles found! \n')  

def main(burn_in, epsilon, max_iters, dataset):
    
    np.seterr(divide='ignore', invalid='ignore')
    start = time.time()
    
    z = sct.norm.ppf(1 - .05 / 2)
    confidence_interval = np.inf
    running_mean, running_2moment, N = 0, 0, 0
    
    if dataset == 'small':
        df = pd.read_csv('AA228Student/workspace/project1/small.csv')
    elif dataset == 'medium':
        df = pd.read_csv('AA228Student/workspace/project1/medium.csv')
    elif dataset == 'large':
        df = pd.read_csv('AA228Student/workspace/project1/large.csv')
    else:
        return None
    
    data = np.matrix(df)   
    i_map = [(i, df.columns[i]) for i in range(len(df.columns))]
    k_map = [(df[i].nunique(), df[i].unique()) for i in df.columns]
    ik_map = [i_map[i] + k_map[i] for i in range(len(k_map))]
   
    values = {}
    for i in range(len(ik_map)):
        values[i] = ik_map[i][3].tolist()
        
    best_parents = {}
    for i in range(len(df.columns)):
        best_parents[i] = []    
    best_score = -np.inf
    
    while(N < burn_in or confidence_interval >= epsilon) and N < max_iters:
        parents = {}
        for i in range(len(df.columns)):
            parents[i] = []
        parents, score = K2(data, parents, values, ik_map)
        if score > best_score:
            best_score = score
            best_parents = parents

        running_mean = (running_mean * N + score) / (N + 1)
        running_2moment = (running_2moment * N + pow(score, 2)) / (N + 1)
        sample_std = math.sqrt(running_2moment - pow(running_mean, 2))
        confidence_interval = z * sample_std / (math.sqrt(N))
        N += 1
    CI = [running_mean - confidence_interval, running_mean + confidence_interval]
    
    finish = time.time()
    print('Generated', N, 'samples in ', round(finish-start,2), ' seconds \n')
    print('95% Confidence Interval: ', CI)
    
    write_gph(dataset, best_parents, ik_map)
    grapherator(dataset, best_parents, ik_map)
    
    return best_parents, best_score


if __name__ == '__main__':
    main(5, .05, 10, 'small')
