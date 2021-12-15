using Documenter

using MaxEntropyGraphs
#push!(LOAD_PATH, "../src/")

ci = get(ENV, "CI", "") == "true"

makedocs(sitename="MaxEntropyGraphs.jl",
         authors="Bart De Clerck",
         format = Documenter.HTML(prettyurls = ci),
         modules=[MaxEntropyGraphs],
         pages = [
            "Home" => "index.md",
            "Models" => "models.md",
            "Higher order metrics" => "derivedquantities.md",
            "GPU acceleration" => "GPU.md"
         ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
if ci
    @info "Deploying documentation to GitHub"
    deploydocs(
        repo = "github.com/B4rtDC/MaxEntropyGraphs.jl.git",
        devbranch = "main"
        )
end