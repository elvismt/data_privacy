# Copyright (C) 2017 Elvis Teixeira
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import numpy as np
import math as m


def date_to_days(date):
    '''Returns the number of days from 0/0/0 to date'''
    d = date.split('/')
    return int(d[1]) + \
           int(d[0]) * 30 + \
           int(d[2]) * 365


def read_data(file_name='diseases.csv'):
    '''Returns the dataframe and the birth date range length'''
    df = pd.read_csv(file_name)
    min_date = None
    max_date = None
    for row in df.iterrows():
        d = date_to_days(row[1][2])
        if min_date is None or d < min_date:
            min_date = d
        if max_date is None or d > max_date:
            max_date = d
    return df, float(max_date - min_date)


def distance(lhs, rhs, date_range=1.0):
    '''Returns the distance between two records'''
    # Gender
    x1 = 0.0 if lhs[1] == rhs[1] else 1.0
    # Birth date
    x2 = float(abs(date_to_days(lhs[2]) - date_to_days(rhs[2]))) / date_range
    # Location (state and city)
    x3 = 0.0
    if lhs[4] != rhs[4]:
        x3 = 1.0
    elif lhs[3] != rhs[3]:
        x3 = 0.5
    # Euclidean distance
    return m.sqrt(x1**2 + x2**2 + x3**2)


def extract_k_group(data, distances, k):
    '''Extracts from mat the first row and it's K-1 nearest neighbors a new matrix
    and returns it together with the new version of mat (without the extracted rows)
    Returns:
       group: the set of K close individuals to anonimize
       reduced_data: the dataset with the ones in 'group' removed
       reduced_distances: the distances matrix with the ones in 'group' removed
    '''
    row = np.copy(distances[0,:])
    group_idxs = np.argsort(row)[:k]
    # Select individuals from dataset
    group = data[group_idxs,:]
    # Create reduced version of data matrix
    reduced_data = np.delete(data, group_idxs, axis=0)
    # Create reduced version of distance matrix
    reduced_distances = np.delete(distances, group_idxs, axis=0)
    reduced_distances = np.delete(reduced_distances, group_idxs, axis=1)
    
    return (group, reduced_data, reduced_distances)


def anonimize_group(group):
    '''Attempts to make all records in group indistinguishabe'''
    
    for i in range(len(group)):
        # Name must vanish
        group[i][0] = '*'
        
        for j in range(i, len(group)):
            # Now check the others to hide in the crowd
            
            if group[i][1] != group[j][1]:
                # Gender is different, suppress
                group[i][1] = '*'
                group[j][1] = '*'
            
            if group[i][4] != group[j][4]:
                # State is different, no need to check city, remove both
                group[i][3] = '*'
                group[i][4] = '*'
                group[j][3] = '*'
                group[j][4] = '*'
            elif group[i][3] != group[j][3]:
                # State is the same but different cities, let's take out just city
                group[i][3] = '*'
                group[j][3] = '*'
            
            d_i = group[i][2].split('/')
            d_j = group[j][2].split('/')
            if d_i[2] != d_j[2]:
                # Year is different, no need to check day or month, remove all
                group[i][2] = '*/*/*'
                group[j][2] = '*/*/*'
            elif d_i[0] != d_j[0]:
                # Month is different, no need to check day, remove month and day
                group[i][2] = '*/*/{}'.format(d_i[2])
                group[j][2] = '*/*/{}'.format(d_j[2])
            elif d_i[1] != d_j[1]:
                # Day is different, remove day
                group[i][2] = '{}/*/{}'.format(d_i[0], d_i[2])
                group[j][2] = '{}/*/{}'.format(d_j[0], d_j[2])
    
    return group


def anonimize_dataset(data, distances, k):
    '''Divides the dataset into groups of K or K+1 and attempts
    to anonimize the set (in the sense of K anonimity)'''
    anonimized_dataset = None
    
    # Since the number of records in the dataset may not be a multiple
    # of K, if we only use K-groups there may be a last group of less
    # than K individuals. To solve this issue we use some K+1 groups
    # to account for the cases of non 'k-multiples'
    k_plus_one_groups = data.shape[0] % k
    k_groups = (data.shape[0] // k) - k_plus_one_groups
    
    # The number of K+1 groups will be equal to the remainder of the
    # division of the number of individuals in the dataset by K. We
    # process those groups first
    for i in range(k_plus_one_groups):
        group, data, distances = extract_k_group(data, distances, k + 1)
        if anonimized_dataset is None:
            anonimized_dataset = anonimize_group(group)
        else:
            anonimized_dataset = np.vstack([anonimized_dataset, anonimize_group(group)])
    # then we process the K groups
    for i in range(k_groups):
        group, data, distances = extract_k_group(data, distances, k)
        if anonimized_dataset is None:
            anonimized_dataset = anonimize_group(group)
        else:
            anonimized_dataset = np.vstack([anonimized_dataset, anonimize_group(group)])
    
    # Check all data was consumed and return the results
    if data.shape[0] != 0:
        raise RuntimeError('not all data consumed')
    return np.array(anonimized_dataset)


def program():
    # Get the dataset into a pandas dataframe and
    # a numpy array (matrix) version
    df, date_range = read_data()
    data_mat = df.as_matrix()
    N = len(df)

    # If there is a saved version of the distances
    # matrixes, use it to save time. Otherwise,
    # compute a new one and save it
    dist_mat = None
    if os.path.exists('distances.npy'):
        dist_mat = np.load('distances.npy')
    else:
        dist_mat = np.reshape([distance(data_mat[i], data_mat[j], date_range) \
            for i in range(N) for j in range(N)], (N, N))
        np.save('distances.npy', dist_mat)
    
    anonimized_datasets = [(k, anonimize_dataset(data_mat, dist_mat, k)) \
        for k in [2, 3, 4, 8, 16, 32]]
    for k, data in anonimized_datasets:
        np.savetxt('result_k{}.csv'.format(k), data, delimiter=',',  fmt="%s")

# Let's get to action
if __name__ == '__main__':
    program()
