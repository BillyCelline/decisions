import numpy as np, scipy as sp, pandas as pd, sklearn as sk
import time
from copy import deepcopy
from sklearn.neighbors import KNeighborsRegressor

small = pd.read_csv('small.csv') 
medium = pd.read_csv('medium.csv')
large = pd.read_csv('large.csv')

def transition_calc(data):
    
    states, actions = data.s.unique(), data.a.unique()
    state_count, action_count = len(states), len(actions)
    
    tmat = np.zeros((state_count, action_count, state_count))
    rmat = np.zeros((state_count, action_count))

    for i in data.values:  
        tmat[i[0]-1, i[1]-1, i[3]-1] += 1
        rmat[i[0]-1, i[1]-1] = i[2]
    tmat /= np.sum(tmat, axis=-1, keepdims=True)
    
    return states, actions, tmat, rmat

def value_iteration(data, iters, discount, verbose):
    
    start = time.time()
    states, actions, tmat, rmat = transition_calc(data)
    state_count, action_count = len(states), len(actions)
    policy = np.zeros((iters, state_count))
    v = np.zeros(state_count)
        
    for i in range(iters):
        
        v_temp = np.zeros(state_count)
        
        for state in range(state_count):
            rewards = {}
            
            if verbose == True:
                print('\ncurrent state:', state, '\ncurrent value:', v[state])
                print('possible states:', np.where(tmat[state][0] !=0)[0],'\n')
            
            total = 0
            for action in range(action_count):
                exp_r = tmat[state][action]@rmat[:,action]
                possible_actions = np.where(tmat[state][action] != 0)[0]
                next_states = np.sum([v[j]*tmat[state][action][j] for j in possible_actions])
                rewards[action] = exp_r + (discount * next_states)

            if verbose == True:
                print('Actions/Rewards:\n', rewards)
            
            reward_max = max(rewards.values())
            v_temp[state] = reward_max

            best_action = [x==reward_max for x in rewards.values()].index(True)
            policy[i][state] = best_action
            
            if verbose == True:
                print('### Optimal Action: ', best_action,'###\n')

        v = v_temp

        
        if verbose == True:
            print('###############\nvalue update:', v, '\n###############')
    
    finish = time.time()
    print('Ran ', i, 'iterations in ', round(finish-start,2), ' seconds \n')

    return v, policy[-1]+1

def write_policy(dataset, policy):
    file = open(dataset + ".policy", "w")
    
    for i in range(len(policy)):
        state = str(int(policy[i])) +"\n"
        file.write(state)

    file.close()

def Q_update(data, Q, alpha, gamma):
    r = 0
    for i in range(len(data)):
        s, a, r, s_prime = data.values[i]
        Q[s-1, a-1] = Q[s-1, a-1] + alpha*(r + gamma*np.max(Q[s_prime - 1]) - Q[s-1, a-1])
    return Q

def Q_approx(data, Q, interval):
    m, n = Q.shape
    knn = KNeighborsRegressor(n_neighbors=interval)
    for i in range(n):
        for j in range(0, m, interval):
            x = np.matrix([[x for x in range(j,j+interval)],
                            np.tile(i, interval)]).reshape(interval,2)
            y = Q[j:j+interval,i]
            knn.fit(x, y) 
            for k in range(interval):
                if len(data[data.s==(j+k)])==0:
                    Q[j+k, i] = knn.predict([[j+k, i]])
    return Q

def Q_learn(data, states, actions, iters, alpha, gamma, interval, strategy):
    
    start = time.time()
    Q, Q_temp = np.zeros((states,actions)), np.zeros((states,actions))
    
    Q = Q_update(data, Q, alpha, gamma)
    if strategy == 'local':
        Q = Q_approx(data, Q, interval)  
    
    diff, i = 1, 0
    while diff > 0.1 and i < iters:
        Q = Q_update(data, Q, alpha, gamma)
        if strategy == 'local':
            Q = Q_approx(data, Q, interval)  
        
        diff = np.sum((Q_temp-Q)**2)
        i += 1
        Q_temp = deepcopy(Q)
        print('i: ',i,'diff: ',diff)
    iters = i
    
    policy = np.argmax(Q,axis=1)+1
    if strategy == 'random':
        for i in range(len(policy)):
            if policy[i] == 1:
                policy[i] = np.random.randint(actions) + 1
    
    finish = time.time()
    print('Ran ', iters, 'iterations in ', round(finish-start,2), ' seconds \n')
    
    return Q, policy

#small test

iters = 1000
discount = 0.95
verbose = False

s_v, s_policy = value_iteration(small, iters, discount, verbose)

#medium test

alpha = .95
gamma = 1

states = 50000
actions = 7

iters = 1000
interval = 10000
strategy = 'random'

m_Q, m_policy = Q_learn(medium, states, actions, iters, alpha, gamma, interval, strategy)
len(m_policy[m_policy!=1])


#large test

alpha = .95
gamma = .95

states = 312020
actions = 9

iters = 1000
interval = 78005
strategy = 'random'

l_Q, l_policy = Q_learn(large, states, actions, iters, alpha, gamma, interval, strategy)
len(l_policy[l_policy!=1])