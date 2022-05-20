from eva import EvaProgram, Input, Output, evaluate
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import timeit
import sys
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph

from random import randrange
import random
from pprint import pprint

SOURCE_NODE = None
SINK_NODE = None
ROW_COUNT = None
dummy_matrix = None
parent = []
run_test = False

# INTERFACE FOR THE COMMUNICATOIN BETWEEN CLIENT AND THE graphanalticprogram
OPERATION_TYPE = None
OPERATION_TYPE_ADD_MAX_FLOW = 'ADD_TO_MAX_FLOW'
OPERATION_TYPE_SUM_TWO_GRAPH = 'SUM_TO_MATRIX'
OPERATION_TYPE_CALCULATE_NEW_INDEX = 'CALCULATE_NEW_INDEX'


call_graphanaltic_program_count = 0 
compiletime = 0
keygenerationtime = 0
encryptiontime = 0 
executiontime = 0
decryptiontime = 0
referenceexecutiontime = 0 
mse = 0 

class Client:
    def __init__(self, graph):
        global ROW_COUNT, SOURCE_NODE, SINK_NODE
        self.row_count = ROW_COUNT
        self.graph = graph  
        self.source = SOURCE_NODE # Source Node for the ford fulkerson algorithm
        self.sink = SINK_NODE #Sink Node for the ford fulkerson algorithm
        self.parent = [-1] * self.row_count

    def BFS(self, s, t, parent):
        
        # Mark all the vertices as not visited
        visited = [False] * (self.row_count)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:
            # Dequeue a vertex from queue and print it
            current_node_index = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for new_node_index in range(0, self.row_count):
                dummy_matrix = [1] * m
                dummy_matrix[0] = current_node_index
                dummy_matrix[1] = self.row_count
                dummy_matrix[2] = new_node_index
                inputs = {'Graph' :  dummy_matrix} # Create a dummy matrix to calculate index of 1D Array from i and j values 
                new_matrix_index = call_graphanaltic_program(inputs, OPERATION_TYPE_CALCULATE_NEW_INDEX) # Call graphanaltic program with corresponding operation type
                new_matrix_index = round(new_matrix_index[0]) # Get the result from the first value and round it since array accesing can be done with int value. 

                new_val = self.graph[new_matrix_index]
                if visited[new_node_index] == False and round(new_val) > 0: # Comparison operation must be done in client side 
                    # If we find a connection to the sink node,
                    # then there is no point in BFS anymore
                    # We just have to set its parent and can return true
                    queue.append(new_node_index)
                    visited[new_node_index] = True
                    parent[new_node_index] = current_node_index
                    if new_node_index == t: # Comparison operation must be done in client side 
                        return True

        # We didn't reach sink in BFS starting
        # from source, so return false
        return False
    
    def run_algorithm(self):
        global m, dummy_matrix, parent
        # This array is filled by BFS and to store path

        max_flow = 0  # There is no flow initially
        # Augment the flow while there is path from source to sink
        while self.BFS(self.source, self.sink, parent):

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = self.sink
            while (s != self.source): # Comparison operation must be done in client side 
                dummy_matrix = [1] * m
                dummy_matrix[0] = parent[s]
                dummy_matrix[1] = self.row_count
                dummy_matrix[2] = s
                inputs = {'Graph' :  dummy_matrix} # Create a dummy matrix for 1D array index calculation. 
                tmp_flow_index = call_graphanaltic_program(inputs, OPERATION_TYPE_CALCULATE_NEW_INDEX)
                tmp_flow_index = round(tmp_flow_index[0]) # Get the result from the first value and round it since array accesing can be done with int value. 

                path_flow = min(path_flow, self.graph[tmp_flow_index]) # min operation must be done in the client side. 
                s = parent[s]

            # Add path flow to overall flow
            
            inputs = {'Graph' :  [max_flow, path_flow] + (m-2) * [0]} # Addition operation can be done in the graphanaltic program since it supports the summation operations 
            max_flow = call_graphanaltic_program(inputs, OPERATION_TYPE_ADD_MAX_FLOW)[0] # Call the function with corresponding operation type

            # update residual capacities of the edges and reverse edges
            # along the path
            v = self.sink
            while (v != self.source): # Comparison operation must be done in client side 
                u = parent[v] 
                dummy_matrix = [0] * m 
                dummy_matrix[u * self.row_count + v] = -1 * path_flow
                dummy_matrix[v * self.row_count + u] = path_flow
                inputs = {'Graph' :  self.graph} # Create a dummy matrix to update the path flows. Both residual capacities of the edges and reverse edges must be updated. 

                new_graph = call_graphanaltic_program(inputs, OPERATION_TYPE_SUM_TWO_GRAPH) #  Call the function with corresponding operation type
                self.graph = new_graph # Update the graph in the client side 
                
                v = parent[v]

        return max_flow 

def generateGraph(n, k, p):
    global SOURCE_NODE, SINK_NODE, ROW_COUNT
    SOURCE_NODE = 0
    SINK_NODE = n-1
    ROW_COUNT = n

    # The graph for the ford-fulkerson algorithm must be directed and connected. 
    # Also there should be a path from source node to sink node.
    # Also there shouldn't be a cycle in the graph for ford-fulkerson algorithm
    # So the graph creation algorithm is modified. 

    ws = nx.DiGraph()
    for i in range(0, n - 1): 
        t = list(range(i + 1, n))
        connected_nodes = random.choices(t, k=max(1, int((n - i) * p)))
        for t in connected_nodes:
            ws.add_edge(i, t)
    return ws


# If there is an edge between two vertices its weight will be a random number otherwise it is zero
# Two dimensional adjacency matrix is represented as a vector
# Assume there are n vertices
# (i,j)th element of the adjacency matrix corresponds to (i*n + j)th element in the vector representations
def serializeGraphZeroOne(GG, last_node, vec_size):
    global ROW_COUNT, parent, run_test

    n = len(GG.nodes())
    graphdict = {}
    if not run_test:
        g = []
        for row in range(n):
            for column in range(n):
                if GG.has_edge(row, column) and row != last_node and column != 0:  # None of the vertics is connected to itself. 
                    weight = randrange(1, 11)  # Assign edge weights randomly selected from 1 to 11
                else:
                    weight = 0
                g.append(weight)
                key = str(row) + '-' + str(column)
                graphdict[key] = [weight]  # EVA requires str:listoffloat
        # EVA vector size has to be large, if the vector representation of the graph is smaller, fill the eva vector with zeros
        for i in range(vec_size - n * n):
            g.append(0.0)

    else:
        graph = [[0, 16, 13, 0, 0, 0],
                [0, 0, 10, 12, 0, 0],
                [0, 4, 0, 0, 14, 0],
                [0, 0, 9, 0, 0, 20],
                [0, 0, 0, 7, 0, 4],
                [0, 0, 0, 0, 0, 0]]
        g = []
        for i in graph:
            for j in i:
                g.append(j)
        for i in(range(vec_size - len(g))):
            g.append(0.0)

    parent = [-1] * ROW_COUNT  
    return g, graphdict


# To display the generated graph
def printGraph(graph, n, m):
    for row in range(n):
        for column in range(n):
            print("{}".format(graph[row * +column]), end='\t')
        print()

    # Eva requires special input, this function prepares the eva input


# Eva will then encrypt them
def prepareInput(n, m):
    input = {}
    GG = generateGraph(n, 3, 0.6)  #  INCREASE THE EDGE POSSIBILITY
    graph, graphdict = serializeGraphZeroOne(GG, n - 1, m)
    input['Graph'] = graph
    return input


# This is the dummy analytic service
# You will implement this service based on your selected algorithm
# you can other parameters using global variables !!! do not change the signature of this function
def graphanalticprogram(graph):    
    # The communication between clientside and graphanalticprogram will be established via theese operation_types. 
    # Each operation type will be set in the client side and checked in the program to perform corresponding operation

    global OPERATION_TYPE, OPERATION_TYPE_ADD_MAX_FLOW, OPERATION_TYPE_SUM_TWO_GRAPH, dummy_matrix
    if OPERATION_TYPE == OPERATION_TYPE_ADD_MAX_FLOW: 
        # Graph is created as [previous_max_flow, added_flow, 0, 0, ...], the result must be previous_max_flow + added_flow.
        # So we shift the matrix by 1 and then we sum original and shifted matrixes.
        reval = graph + (graph << 1) 
    elif OPERATION_TYPE == OPERATION_TYPE_SUM_TWO_GRAPH:
        # We created a dummy matrix which contains a sub matrix (n*n).
        # n*n matrix contains the addition values for resudual graphs. 
        reval = graph + dummy_matrix
    elif OPERATION_TYPE == OPERATION_TYPE_CALCULATE_NEW_INDEX: 
        # Graph is created as [ith_index, row_size, jth_index, 0, 0, ...] for the calculation of the index of the element in the 1D array.
        # The index equals to ith_index + row_size + jth_index.
        # So we multiply the graph with 1 time shifted graph and sum the result with the two times shifted graph 
        reval = graph * (graph << 1) + (graph << 2)
    else:
        # If no operation type is given, just return the value        
        reval = graph 
        
    return reval


# Do not change this
#  the parameter n can be passed in the call from simulate function
class EvaProgramDriver(EvaProgram):
    def __init__(self, name, vec_size=4096, n=4):
        self.n = n
        super().__init__(name, vec_size)

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)


def call_graphanaltic_program(inputs, operation_type=None):
    # With given operation type and inputs, create a Eva Program, encrypt the input and call the function with the encrypted input
    # The encrypted output should be decrypted with the private key and return the results.  

    global m, n, OPERATION_TYPE, config
    global compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, call_graphanaltic_program_count

    call_graphanaltic_program_count += 1 # Count how many times the function is called 

    OPERATION_TYPE = operation_type

    graphanaltic = EvaProgramDriver("graphanaltic", vec_size=m, n=n)
    with graphanaltic:
        graph = Input('Graph')
        reval = graphanalticprogram(graph)
        Output('ReturnedValue', reval)

    prog = graphanaltic
    prog.set_output_ranges(30)
    prog.set_input_scales(30)

    start = timeit.default_timer()
    compiler = CKKSCompiler(config=config)
    compiled_multfunc, params, signature = compiler.compile(prog)
    compiletime += (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    public_ctx, secret_ctx = generate_keys(params)
    keygenerationtime = (timeit.default_timer() - start) * 1000.0 #ms
    
    start = timeit.default_timer()
    encInputs = public_ctx.encrypt(inputs, signature)
    encryptiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    encOutputs = public_ctx.execute(compiled_multfunc, encInputs)
    executiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    outputs = secret_ctx.decrypt(encOutputs, signature)            
    decryptiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    reference = evaluate(compiled_multfunc, inputs)
    referenceexecutiontime = (timeit.default_timer() - start) * 1000.0 #ms
    
    c = 0
    output = None
    for key in outputs:
        output = outputs[key]
        break
    
    return output


def simulate(n_value):
    global client, m, n, dummy_matrix, config
    global client, m, n, dummy_matrix, config, compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse
    
    n = n_value
    m = 4096 * 4
    
    config = {}
    config['warn_vec_size'] = 'false'
    config['lazy_relinearize'] = 'true'
    config['rescaler'] = 'always'
    config['balance_reductions'] = 'true'
    
    inputs = prepareInput(n, m)
    client = Client(inputs['Graph'])

    result = client.run_algorithm()
   
    mse = (result - round(result)) **2  # since CKKS does approximate computations, this is an important measure that depicts the amount of error

    return result, compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse


if __name__ == "__main__":
    simcnt = 3  # The number of simulation runs, set it to 3 during development otherwise you will wait for a long time
    # For benchmarking you must set it to a large number, e.g., 100
    # Note that file is opened in append mode, previous results will be kept in the file
    resultfile = open("results.csv", "a")  # Measurement results are collated in this file for you to plot later on
    resultfile.write("NodeCount,PathLength,Result,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")
    resultfile.close()
    if len(sys.argv) > 1:
        test_case = sys.argv[1]
        if test_case == 'run_test':
            run_test = True
            result, compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse = simulate(6)
            res = str(6) + "," + str(0) + "," + str(result) + "," + str(compiletime) + "," + str(keygenerationtime) + "," + str(encryptiontime) + "," + str(executiontime) + "," + str(decryptiontime) + "," + str(referenceexecutiontime) + "," + str(mse) + ',' + '23' + "\n"
            pprint(dict(zip(['NodeCount','PathLength','Result','CompileTime','KeyGenerationTime','EncryptionTime','ExecutionTime','DecryptionTime','ReferenceExecutionTime','Mse','Expected Result'], res.split(','))))
    if not run_test:
        for nc in [36, 64, 4]:  # Node counts for experimenting various graph sizes
            n = nc
            resultfile = open("results.csv", "a")
            for i in range(simcnt):
                # Call the simulator
                result, compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse = simulate(n)
                res = str(n) + "," + str(i) + "," + str(result) + "," + str(compiletime) + "," + str(keygenerationtime) + "," + str(encryptiontime) + "," + str(executiontime) + "," + str(decryptiontime) + "," + str(referenceexecutiontime) + "," + str(mse) + "\n"
                pprint(dict(zip(['NodeCount','PathLength','Result','CompileTime','KeyGenerationTime','EncryptionTime','ExecutionTime','DecryptionTime','ReferenceExecutionTime','Mse',], res.split(','))))
                resultfile.write(res)
                
            resultfile.close()

