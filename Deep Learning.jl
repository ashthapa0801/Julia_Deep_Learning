## Deep Learning with Flux.ij
## Data munging
using Flux, MLDatasets
using CSV, DataFrames, Plots
using ImageShow

# Load training data (images, labels)
x_train, y_train = MLDatasets.MNIST.traindata()
# Load test data (images, labels)
x_test, y_test = MLDatasets.MNIST.testdata()
# Convert grayscale to float
x_train = Float32.(x_train)
# Create labels batch
y_train = Flux.onehotbatch(y_train, 0:9)

## Create the model
# 4 layers, 1st layer containing 784 i/ps and 256 o/ps
# 2nd layer containing 256 i/ps and 10 o/ps
# 3rd layer containing 128 i/ps and 10 o/ps
# 4th layer is the Softmax activaiton layer
model = Chain(
    Dense(784, 256, relu),
    Dense(256, 10, relu)
    Dense(128, 10, relu), softmax
)

# Define Loss function
loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)

# Define Optimizer
optimizer = ADAM(0.0001)

## training
parameters = params(model)
# flatten() function converts array 28x28x60000 into 784x60000 (28*28x60000)
train_data = [(Flux.flatten(x_train), Flux.flatten(y_train))]
# Range in loop can be used smaller
for i in 1:400  # 400 Epoch
    Flux.train!(loss, parameters, train_data, optimizer)
end

# Check Results
test_data = [(Flux.flatten(x_test), y_test)]
accuracy = 0    # Set accuracy to zero
for i in 1:length(y_test)
    if findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
        accuracy = accuracy + 1
    end
end

# Results of test
println(accuracy / length(y_test))