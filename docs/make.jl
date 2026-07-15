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
                            "Which model when?" => "model_selection.md",
                            "UBCM" => "models/UBCM.md",
                            "DBCM" =>  "models/DBCM.md",
                            "RBCM" =>  "models/RBCM.md",
                            "BiCM" =>  "models/BiCM.md",
                            "UECM" =>  "models/UECM.md",
                            "CReM" =>  "models/CReM.md",
                            "DCReM" => "models/DCReM.md",
                            "CRWCM" => "models/CRWCM.md"
                            ],
            "Metrics" => Any["metrics.md",
                             "Analytical" => "exact.md", 
                             "Simulation" => "simulated.md"],
            "Performance and scalability" => "performance.md",
            "API" => Any[   "Shared" =>"API/API.md",
                            "UBCM" => "API/API_UBCM.md",
                            "DBCM" => "API/API_DBCM.md",
                            "RBCM" => "API/API_RBCM.md",
                            "BiCM" => "API/API_BiCM.md",
                            "UECM" => "API/API_UECM.md",
                            "CReM" => "API/API_CReM.md",
                            "DCReM" => "API/API_DCReM.md",
                            "CRWCM" => "API/API_CRWCM.md",]
                            ],
         doctest=false,
         checkdocs=:exports,   # only require exported symbols in the manual (internal helpers like `softplus` are fine)
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
