using Documenter
#push!(LOAD_PATH, joinpath(pwd(),".."))
using MaxEntropyGraphs
#using Graphs

ci = get(ENV, "CI", "") == "true"

makedocs(sitename="MaxEntropyGraphs.jl",
         authors="Bart De Clerck",
         format = Documenter.HTML(prettyurls = ci),
         modules=[MaxEntropyGraphs],
         pages = [
            "Home" => "index.md",
            "Models" => Any["models.md",
                            "UBCM" => "models/UBCM.md",
                            "DBCM" => "models/DBCM.md"],
            "Metrics" => Any["exact.md", "simulated.md"],
            "Helper functions" => "utils.md",
            "GPU acceleration" => "GPU.md",
            "API" => "API.md"
         ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
if ci
    @info "Deploying documentation to GitHub"
    deploydocs(
        repo = "github.com/B4rtDC/MaxEntropyGraphs.jl.git",
        devbranch = "2023rework"
        )
end