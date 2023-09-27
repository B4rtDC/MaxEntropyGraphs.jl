using Documenter


#push!(LOAD_PATH, joinpath(pwd(),".."))
LOCAL = true
if LOCAL
    using Pkg
    # Pkg.add(url="/Users/bart/Documents/Stack/PhD/Coding (experiments)/MaxEntropyGraphs.jl/", rev="2023rework")
    Pkg.update("MaxEntropyGraphs")
    using MaxEntropyGraphs
else
    using MaxEntropyGraphs
end

# to give all docstrings access to the package, we need to import it
DocMeta.setdocmeta!(MaxEntropyGraphs, :DocTestSetup, :(using MaxEntropyGraphs); recursive=true)


# check if we are running on CI
ci = get(ENV, "CI", "") == "true"
const buildpath = haskey(ENV, "CI") ? ".." : "" # https://github.com/JuliaDocs/Documenter.jl/issues/921

# makedocs will run all docstrings in the package
makedocs(sitename="MaxEntropyGraphs.jl",
         authors="Bart De Clerck",
         format = Documenter.HTML(prettyurls = ci),
         modules=[MaxEntropyGraphs],
         pages = [
            "Home" => "index.md",
            "Models" => Any["models.md",
                            "UBCM" => "models/UBCM.md",
                            "DBCM" => "models/DBCM.md"
                            ],
            "Metrics" => Any["metrics.md",
                             "Analytical" => "exact.md", 
                             "Simulation" => "simulated.md"],
            "GPU acceleration" => "GPU.md",
            "API" => "API.md"
         ]#,
         #doctest=false
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