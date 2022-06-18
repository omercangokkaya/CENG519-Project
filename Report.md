Ford-Fulkerson Algorithm for Maximum Flow Problem is implemented in
privacy-preserving manner using homomorphic encryption. The effect of
the using Homomorphic Encryption is analyzed and reported in the report.
The most important finding are that the size of the graph affects the
computation time in algorithm and the result of the encrypted approach
is not different from the result of the actual algorithm.

# Introduction

The aim of the project is implementing Ford-Fulkerson Algorithm for
Maximum Flow Problem Implementation with Homomorphic Encryption(HE).
Although implementing the Ford-Fulkerson Algorithm is easy task since it
is a well defined and already implemented algorithm, it is not easy to
implement it in a privacy-preserving manner using homomorphic encryption
Microsoft Simple Encrypted Arithmetic Library(SEAL) library since the
SEAL library has own computation limitations.\
In my approach, there are two modules which are the client module in
which the client makes operations in his/her local machine and the HE
module in which encrypted operations are made with HE. Most of the
arithmetic operations in the algorithm are done with the homomorphic
encryption in HE module while other operations that are not supported by
HE module are handled in the Client side. Furthermore, I have
implemented an interface for the client module to interact with the HE
module in order to ease the implementation .

## Ford-Fulkerson Algorithm for Maximum Flow Problem Implementation

The Ford-Fulkerson algorithm implemented in
GeeksforGeeks [@GeeksforGeeks] is given as follows:

    # Python program for implementation
    # of Ford Fulkerson algorithm
    from collections import defaultdict
     
    # This class represents a directed graph
    # using adjacency matrix representation
    class Graph:
     
        def __init__(self, graph):
            self.graph = graph  # residual graph
            self. ROW = len(graph)
            # self.COL = len(gr[0])
     
        '''Returns true if there is a path from source 's' to sink 't' in
        residual graph. Also fills parent[] to store the path '''
     
        def BFS(self, s, t, parent):
     
            # Mark all the vertices as not visited
            visited = [False]*(self.ROW)
     
            # Create a queue for BFS
            queue = []
     
            # Mark the source node as visited and enqueue it
            queue.append(s)
            visited[s] = True
     
             # Standard BFS Loop
            while queue:
     
                # Dequeue a vertex from queue and print it
                u = queue.pop(0)
     
                # Get all adjacent vertices of the dequeued vertex u
                # If a adjacent has not been visited, then mark it
                # visited and enqueue it
                for ind, val in enumerate(self.graph[u]):
                    if visited[ind] == False and val > 0:
                          # If we find a connection to the sink node,
                        # then there is no point in BFS anymore
                        # We just have to set its parent and can return true
                        queue.append(ind)
                        visited[ind] = True
                        parent[ind] = u
                        if ind == t:
                            return True
     
            # We didn't reach sink in BFS starting
            # from source, so return false
            return False
                 
         
        # Returns the maximum flow from s to t in the given graph
        def FordFulkerson(self, source, sink):
     
            # This array is filled by BFS and to store path
            parent = [-1]*(self.ROW)
     
            max_flow = 0 # There is no flow initially
     
            # Augment the flow while there is path from source to sink
            while self.BFS(source, sink, parent) :
     
                # Find minimum residual capacity of the edges along the
                # path filled by BFS. Or we can say find the maximum flow
                # through the path found.
                path_flow = float("Inf")
                s = sink
                while(s !=  source):
                    path_flow = min (path_flow, self.graph[parent[s]][s])
                    s = parent[s]
     
                # Add path flow to overall flow
                max_flow +=  path_flow
     
                # update residual capacities of the edges and reverse edges
                # along the path
                v = sink
                while(v !=  source):
                    u = parent[v]
                    self.graph[u][v] -= path_flow
                    self.graph[v][u] += path_flow
                    v = parent[v]
     
            return max_flow
     
      

# Background and Related Work

## Background

Two important concepts need to be explained. The first one is the
Ford-Fulkerson Algorithm for Maximum Flow Problem which can be defined
as: Given a graph which represents a flow network where every edge has a
capacity. Also given two vertices source 's' and sink 't' in the graph,
find the maximum possible flow from s to t with following constraints:

1.  Flow on an edge does not exceed the given capacity of the edge.

2.  Incoming flow is equal to outgoing flow for every vertex except s
    and t. [@GeeksforGeeks].

The other one is the Microsoft Simple Encrypted Arithmetic Library(SEAL)
library which has been available since 2015 and makes performing
computations on encrypted data possible with providing an opportunity to
offer greater security assurances to customers using and storing their
personal information in the cloud. [@SEAL]. Although, SEAL library
enables us to work with encrypted data, it has some limitations and it
only supports arithmetic operations such that addition, left shift,
right shift and negation. Due to this limitation most of the operations
which contain array accessing or comparison in the naive Ford-Fulkerson
algorithm cannot be handled by homomorphic encryption.

## Related Work

The sample usage of the Microsoft SEAL library and example
implementation are given in the microsoft/SEAL github page
 [@SEALGithub]. Although they give an example usage of the library in
the repository, the operations in the example are much simpler than the
operations in our task. However, it is recommended to check the
repository since it provides information about the basic operations and
how to use the library.

# Results and Discussion

## Methodology

Ford-Fulkerson Algorithm implementation(naive algorithm) in
GeeksforGeeks [@GeeksforGeeks] are used with some modifications in the
project. These modifications are:

-   Implemented algorithm in GeeksforGeeks takes NxN 2D array as an
    input however in my implementation the homomorphic encryption
    requires Nx1 1D array as input. All matrix accessing operations in
    the naive algorithm are replaced. For example, assume that we have a
    2D array A with row size n and we want to access the element in the
    i'th row j'th column. In the naive approach we can access it by
    A\[i\]\[j\] while in my algorithm we can access it by A\[i\*n+j\].

-   Arithmetic operations in the naive algorithm are replaced by the
    Homomorphic Encryption calls. An interface is created to handle
    different arithmetic operations.

### Graph creation & representation

To run ford-fulkerson algorithm a valid graph is needed which should
have some requirements:

-   The graph should be weighted directed graph in which edge weights
    represent the capacity of the edge

-   There should be a source edge (S) to which there should be no
    incoming edge

-   There should be a sink edge (T) from which there should be no
    outgoing edge

-   There should be at least one path from s to t

### Interface for Client - HE module communication 

The Ford-Fulkerson algorithm's base is written in the client side of the
project however for different arithmetic operations it is needed to
write a interface for client-HE module communication. Since we are not
allowed to change the function definition of the graphanalticprogram in
the project, some variables are needed to be passed to the function with
global variables. A global OPERATION_TYPE variable is created for
passing operation type to the HE module. In the client side of the
application the OPERATION_TYPE variable is set and in the HE module the
variable value is checked to perform corresponding operation. There are
3 different operation types.

-   OPERATION_TYPE_ADD_MAX_FLOW: In the naive algorithm max flow in the
    graph should be updated after each iteration and this operation can
    be done with HE module. Assume that we have a previous max_flow
    value and we want to add a new_path_flow value to the previous one
    and get a new max_flow value. To handle this operation first an
    array is created with size of M (size of the input array to be
    passed to the HE module) and the first element of the array is set
    to previous max_flow value and the second element of the array is
    set to new_path_flow. This array is encrypted on the client side and
    passes the graphanalticprogram as a parameter. In the
    graphanalticprogram, input array and one shifted of the input array
    is summed and the summation of the array is returned to the client
    side. After that, the client side decrypts the output of the
    graphanalticprogram and gets the first element of the array as new
    max_flow.

-   OPERATION_TYPE_SUM_TWO_GRAPH: In the naive algorithm some edges
    between two vertices in the graph should be updated by flow which
    goes over them and this operation can be done with HE module. Assume
    that we have a residual graph **G** and we want to add a constant
    value **C** to the element in the index of (i1, j1) and subtract a
    constant number **C** from the element in the index of (i2, j2). To
    handle this operation, first a new dummy array **A** with all 0's is
    created with size of M (size of the input array to be passed to the
    HE module) and the element at the index of (i1, j1) is set to C
    value while the element at the index of (i2, j2) is set to -C value.
    The residual graph **G** is encrypted in the client side and pass
    the graphanalticprogram as a parameter. The dummy array **A** is
    passed to the graphanalticprogram with global variable. In the
    homomorphic encryption modele graph G and graph A is summed and the
    encrypted result is returned to the client side. After that in the
    client side the output is decrypted and residual graph **G** is
    replaced with the output.

-   OPERATION_TYPE_CALCULATE_NEW_INDEX: In the naive algorithm there are
    lots of index calculation operations. For example, accessing the
    element in the index of (i,j), the following calculation should be
    made: i\*(size_of_row) + j. Since this is an arithmetic operation,
    this operation can be handled in the HE module. To handle this
    operation first a array is created with size of M (size of the input
    array to be passed to the HE module) and the first element of the
    array is set to **i** value, the second element of the array is set
    to **size_of_row** and the third element of the array is set to
    **j**. This array is encrypted in the client side and pass the
    graphanalticprogram as a parameter. In the graphanalticprogram,
    input array and one shifted of the input array is multiplied to find
    i \* i\*(size_of_row) and the result is summed with the two times
    shifted of the input array to find i\*(size_of_row) + j and the
    result is returned to the client side. After that, the client side
    decrypts the output of the graphanalticprogram and gets the first
    element of the array as a calculated index.

## Results

Several experiments with different configurations are made in the
project and the only configuration parameter is the node count which is
chosen as 8, 10 and 12. Also for each node count 100 different
simulations are made. The main experiment which contains 300 hundred
scenarios is done for analyzing the time cost of different SEAL
operations. There are 5 different operations which are:

-   Compile Time: For the given graphanaltic program, the time passes
    while CKKSCompiler is compiling the program with given vector size
    and configurations in the client side.

-   Key Generation Time: The time passes during Public key and secret
    key generation in the client side. The secret context should be held
    on only the client and should not be shared. The public key is used
    in the encryption operation while the private key is used in the
    decryption operation.

-   Encryption Time: The time passes while encrypting given input with
    the public key of the client. The encrypted data will be used in the
    execution step.

-   Execution Time: The time passes while running the compiled
    graphanaltic program in the Homomorphic Encryption module and
    retrieving the encrypted output from the HE module.

-   Decryption Time: The time passes during the decryption of the
    encrypted output with the private key of the client.

![Node Count vs Compile Time/ Key Generation Time/ Encryption Time/
Execution Time/ Decryption
Time](Project_Nodecount_vs_durations.png){#fig:fig1 width="\\textwidth"}

The Figure [1](#fig:fig1){reference-type="ref" reference="fig:fig1"}
shows the average total time spent on different operations. Average
compile times are much less then the other operations and average
compile times are about 20, 50 and 100 seconds for node counts 8, 10, 12
respectively while Key generation time takes much more time than the
encryption, execution and decryption times. Moreover, when the node
counts are increased, all operations' duration are increased since the
number of graphanaltic program calls are increased. For example, for
node counts 8, 10 and 12, the total number of graphanaltic program calls
are 100, 400 and 900. When we normalize the duration with respect to the
number of program calls, we can get closer values for each node counts.
Furthermore, as shown in the Figure [2](#fig:fig2){reference-type="ref"
reference="fig:fig2"} running an experiment with the same inputs without
using homomorphic encryption takes much less time than the experiment
with HE module.\

![Node Count vs Reference Execution Time without Homomorphic
Encryption.png](Node Count vs Reference Execution Time without Homomorphic Encryption.png){#fig:fig2
width="\\textwidth"}

The second experiment shows the MSE [@MSE] between the result of the
graphanaltic program with homomorphic encryption and the result of the
naive Ford-Fulkerson Algorithm for Maximum. As shown in the Figure
[3](#fig:fig3){reference-type="ref" reference="fig:fig3"}, the mean
squared error is very close to the 0 in different experiments with
different node counts. The difference between expected result and
graphanaltic program's result is occurred since CKKS does an approximate
computations [@EVAApprox].\
Finally, all experiments are run in pyhon3.8 with docker engine 19.03
and all the tests are made with 2,3 GHz 8-Core Intel Core i9, 32 GB 2667
MHz DDR4 MacBook Pro. For the detailed information you can check the
github repository [@ProjectGithub] of the project.

![Node Count vs MSE.png](Project_MSE_Node_Count.png){#fig:fig3
width="\\textwidth"}

## Discussion

As mentioned in the in the introduction, most of the operations in the
Ford-Fulkerson Algorithm cannot be done on encrypted data in Homomorphic
Encryption module so most of the operations such as comparison or
iterations are handled in the client side and this is caused by the
limitation of the SEAL library. Moreover, when encrypted operation is
required on the client side, for each operation I create a new
CKKSCompiler, encryption keys and call the graphanalticprogram. I could
not find a way to create only one EVAL program and call the same program
when needed so it takes too much time to create multiple EVAL programs.
Finally, using Homomorphic Encryption with algorithms which contain more
arithmetic operations is recommended.

# Conclusion

Ford-Fulkerson Algorithm for Maximum Flow problem Implementation with
Homomorphic Encryption(HE) is implemented in the scope of this report.
It is clear to see that the running time of the algorithm is related to
the node count. Furthermore, CKKS's approximate computations do not have
significant effect on the result of the algorithm so the effect can be
minimized by rounding the result to the closest integer value.
