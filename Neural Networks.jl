## Neural Networks
import Pkg; Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaComputing/JuliaAcademyData.jl"))
using JuliaAcademyData; activate("Deep learning with Flux")

# Multiple O/P models
# Distingushing between apples, bananas and grapes

# Load Data
using CSV, DataFrames, Flux, Plots
# Load apple data in CSV.read for each file
apples1 = DataFrame(CSV.File(datapath("data/Apple_Golden_1.dat"), delim='\t', allowmissing=:none, normalizenames=true))
apples2 = DataFrame(CSV.File(datapath("data/Apple_Golden_2.dat"), delim='\t', allowmissing=:none, normalizenames=true))
apples3 = DataFrame(CSV.File(datapath("data/Apple_Golden_3.dat"), delim='\t', allowmissing=:none, normalizenames=true))

# And then concatenate them all together
apples = vcat(apples1, apples2, apples3)
bananas = DataFrame(CSV.File(datapath("data/Banana.dat"), delim='\t', allowmissing=:none, normalizenames=true))

grapes1 = DataFrame(CSV.File(datapath("data/Grape_White.dat"), delim='\t', allowmissing=:none, normalizenames=true))
grapes2 = DataFrame(CSV.File(datapath("data/Grape_White_2.dat"), delim='\t', allowmissing=:none, normalizenames=true))
grapes = vcat(grapes1, grapes2)

# Extract out the features and construct the corresponding labels
x_apples  = [ [apples[i, :red], apples[i, :blue]] for i in 1:size(apples, 1) ]
x_bananas  = [ [bananas[i, :red], bananas[i, :blue]] for i in 1:size(bananas, 1) ]
x_grapes = [ [grapes[i, :red], grapes[i, :blue]] for i in 1:size(grapes, 1) ]

# Concatenate
xs = vcat(x_apples, x_bananas, x_grapes)
ys = vcat(fill([1,0,0], size(x_apples)),
          fill([0,1,0], size(x_bananas)),
          fill([0,0,1], size(x_grapes)))
# ### One-hot vectors
# Recall:
#
# <img src="https://raw.githubusercontent.com/JuliaComputing/JuliaAcademyData.jl/master/courses/Deep%20learning%20with%20Flux/data/fruit-salad.png" alt="Drawing" style="width: 300px;"/>
# `Flux.jl` provides an efficient representation for one-hot vectors, using advanced features of Julia so that it does not actually store these vectors, which would be a waste of memory; instead `Flux` just records in which position the non-zero element is. To us, however, it looks like all the information is being stored:
using Flux: onehot

onehot(2, 1:3)

# Concatenate
ys = vcat(fill(onehot(1, 1:3), size(x_apples)),     # element 1 is false
          fill(onehot(2, 1:3), size(x_bananas)),    # element 2 is false
          fill(onehot(3, 1:3), size(x_grapes)))     # element 3 is false


## The core algorithm from the previous lecture
model = Dense(2, 1, σ)
L(x,y) = Flux.mse(model(x), y)
opt = SGD(params(model))
Flux.train!(L, zip(xs, ys), opt)

for _ in 1:100  # Train for 100 epoch
    Flux.train!(L, zip(xs, ys), opt)
end

## Visualising
using Plots
plot()

contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[1], levels=[0.5, 0.51], color = cgrad([:blue, :blue]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[2], levels=[0.5,0.51], color = cgrad([:green, :green]))
contour!(0:0.01:1, 0:0.01:1, (x,y)->model([x,y]).data[3], levels=[0.5,0.51], color = cgrad([:red, :red]))

scatter!(first.(x_apples), last.(x_apples), m=:cross, label="apples", color = :blue)
scatter!(first.(x_bananas), last.(x_bananas), m=:circle, label="bananas", color = :green)
scatter!(first.(x_grapes), last.(x_grapes), m=:square, label="grapes", color = :red)