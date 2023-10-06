using Documenter
using Pkg




# check if we are running on CI
ci = get(ENV, "CI", "") == "true"
@info "CI status: $ci"
const buildpath = haskey(ENV, "CI") ? ".." : "" # https://github.com/JuliaDocs/Documenter.jl/issues/921 for images

# activate the package environment
if !ci
    Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "..")))
    Pkg.instantiate()
end

using MaxEntropyGraphs

# to give all docstrings access to the package, we need to import it
DocMeta.setdocmeta!(MaxEntropyGraphs, :DocTestSetup, :(using MaxEntropyGraphs); recursive=true)

# makedocs will run all docstrings in the package
makedocs(sitename="MaxEntropyGraphs.jl",
         authors="Bart De Clerck",
         format = Documenter.HTML(prettyurls = ci),
         modules=[MaxEntropyGraphs],
         pages = [
            "Home" => "index.md",
            "Models" => Any["models.md",
                            "UBCM" => "models/UBCM.md",
                            "DBCM" =>  "models/DBCM.md",
                            "BiCM" =>  "models/BiCM.md"
                            ],
            "Metrics" => Any["metrics.md",
                             "Analytical" => "exact.md", 
                             "Simulation" => "simulated.md"],
            "GPU acceleration" => "GPU.md",
            "API" => Any[   "Shared" =>"API/API.md",
                            "UBCM" => "API/API_UBCM.md",
                            "DBCM" => "API/API_DBCM.md",
                            "BiCM" => "API/API_BiCM.md",]
                            ],
         doctest=true,
         build=joinpath(dirname(@__FILE__), "build")
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
if ci
    @info "Deploying documentation to GitHub"
    deploydocs(
        repo = "github.com/B4rtDC/MaxEntropyGraphs.jl.git"
        )
end
