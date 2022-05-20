# Project Information
This project is created for CENG519 course. In this project, Ford-Fulkerson Algorithm for Maximum Flow Problem is implemented by using Microsoft SEAL and EVA libraries. Required packages will be installed by dockerfile. 

In the project, we will talk about two entity. 

One of them is graphanalticprogram which has capability of doing limited opeartions such that addition, right and left shifting on arrays, multiplication etc. This program is used in EvaProgram. For the details you can check the folowing link. Microsoft SEAL and Microsoft EVA: https://github.com/microsoft/EVA

The second entity is the client -a python class- which can do any operation that python do. In the client entity, we do the most of the jobs for the Ford-Fulkerson algorithm since the algorithm's core contains iteration over array, comparison opeartions, assignment operations. However, most of the aritmetic operations such that shifting, summation, multiplication etc is not done in the client side.

While implementing the algorithm, firstly we need an interface for graphanalticprogram to talk with the client side. This interface is provided by some constant values and a OPERATION_TYPE variable. These constants are:
    * OPERATION_TYPE_ADD_MAX_FLOW (graphanalticprogram sums first two values of the array and retruns the result at the first index of the array.)
    * OPERATION_TYPE_SUM_TWO_GRAPH (graphanalticprogram sums the given graph with the dummy graph which is created in the client side and pass via global parameters to the function. graphanalticprogram returns the summation of these two arrays(graphs) as a result.)
    * OPERATION_TYPE_CALCULATE_NEW_INDEX (graphanalticprogram calculates the 1D array index by given inputs in the array. The M*M array's first three element is set as (ith_index, row_size, jth_index) in the client side. This array is passed the graphanalticprogram as enrypted. graphanalticprogram computes the following equation (ith_index \* row_size + jth_index) and returns the value. )

After creating the interface, I implemented the ford-fulkerson algorithm in the client class. The main part of the algorithm is handled in the run_algorithm function. In this function I compute all possible flows from source node to sink node and update the residual capacities on the graph. For the details of the algorith you can check the https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/ page. In this functioni there are some aritmetic computations for example calculating the 1D array index by i and j values. This operation is handled in the graphanaltic_program. Also, there is a operation in which I calculated the max_flow, this one is also handled in the graphanaltic_program. Moreover, updating the residual capacities of the edges and reverse edges is also an arithmetic operation so it is handled in the graphanaltic_program. 

We will do the flow calculations until there is no usable path from source node to sink node in the graph. This check will be done with BFS function. BFS function is implemented clint Class too. This function basicly tries to find a path from source to sink with at least 1 flow. This function is called BFS since the path creation operation is started from the source node and in each step one of the child of the current node will be added to the path until the current node is sink node. This operation is basically breadth first search. 

Let me briefly talk about input creation. Input is not created randomly. We need to create a flow network for the given task. In graph theory, a flow network is defined as directed graph G = (V, E) constrained with a function c, which bounds each edge e with a non-negative integer value which is known as capacity of the edge e with two additional vertices defined as source S and sink T. Source vertex should have all outgoing edges and no incoming edges, while sink vertex should have all incoming edges and no outgoing edge. 


# Running the example

Run random cases: 
```
python3 519ProjectTemplate/fhe_template_project.py
```

Run specific test case created by developper
```
python3 519ProjectTemplate/fhe_template_project.py run_test
```

Graph in the test case is as following and taken from https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
graph = [[0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0]]
result = 23 

HAVE FUN.