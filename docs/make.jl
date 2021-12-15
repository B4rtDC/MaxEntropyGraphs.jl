using Documenter


makedocs(sitename="MaxEntropyGraphs.jl",
         pages = [
            "Home" => "index.md",
            "Models" => "models.md",
            "Higher order metrics" => "derivedquantities.md",
            "GPU acceleration" => "GPU.md"
         ],
         format = Documenter.HTML(prettyurls = false)
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/B4rtDC/MaxEntropyGraphs.jl.git",
    devbranch = "main"
)