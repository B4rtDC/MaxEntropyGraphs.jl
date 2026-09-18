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
                            "DBiCM" => "models/DBiCM.md",
                            "UECM" =>  "models/UECM.md",
                            "DECM" =>  "models/DECM.md",
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
                            "DBiCM" => "API/API_DBiCM.md",
                            "UECM" => "API/API_UECM.md",
                            "DECM" => "API/API_DECM.md",
                            "CReM" => "API/API_CReM.md",
                            "DCReM" => "API/API_DCReM.md",
                            "CRWCM" => "API/API_CRWCM.md",]
                            ],
         doctest=true,
         # Doctests print maximum-likelihood parameters, likelihoods and information criteria, which
         # are floating-point results of an iterative solve: their last digits depend on the BLAS, the
         # platform and the optimiser version, and drift with any of them. Pinning all 17 digits made
         # the examples fail on machines other than the one they were written on, which is why
         # `doctest` had been switched off entirely - leaving 18 genuinely broken examples invisible
         # for as long as it was off.
         #
         # Comparing a fixed number of leading decimals instead makes the check portable. Note the one
         # way truncation can still bite: two values either side of a digit boundary truncate
         # differently however close they are (0.39842998 and 0.39842997 came from the same solve on
         # two platforms). The risk scales with drift / 10^-k, and the drift on these solves is ~1e-9
         # rather than ULP-level, so k = 5 leaves roughly four orders of margin. Examples whose value
         # is genuinely uncertain beyond that are rounded at the call site instead, and none print a
         # whole parameter vector any more - each number there was another chance to straddle.
         doctestfilters=[r"(\d+\.\d{5})\d+" => s"\1"],
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
