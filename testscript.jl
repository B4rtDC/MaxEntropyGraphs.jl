using Graphs
using MaxEntropyGraphs

g = erdos_renyi(50,0.1)
model = UBCM(g)

mymodel = DBCM(erdos_renyi(50,0.1, is_directed=true))