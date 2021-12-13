using BenchmarkTools
using fastmaxent
using LightGraphs


G = LightGraphs.static_scale_free(100,2000,2.4)
large_model = UBCM(G)
compact_model = UBCMCompact(G)
res_large = solve!(large_model)
res_compact = solve!(compact_model)

suite = BenchmarkGroup()
suite["large"] = @benchmarkable solve!($large_model)
suite["compact"] = @benchmarkable solve!($compact_model)
tune!(suite)
results = run(suite)