using Documenter


@info @__FILE__, @__DIR__
push!(LOAD_PATH, joinpath(@__DIR__,"../src/"))
using Pkg
Pkg.activate(joinpath(@__DIR__,"../"))
using MaxEntropyGraphs


makedocs(sitename="MaxEntropyGraphs.jl",
         pages = [
            "Home" => "index.md",
            "An other page" => "anotherPage.md",
            "Models" => "models.md",
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