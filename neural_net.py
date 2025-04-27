import numpy as np

# modify this block of stuff
layer_node_sizes = [2, 10, 5, 1]
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
answers = np.array([[0], [1], [1], [0]])
learning_rate = 0.005

all_node_values = []
all_activated_values = [inputs]

def activation(layer_node_values):
  # sigmoid
  return 1 / (1 + np.exp(-layer_node_values))

def get_activation_derivative(values):
  x = activation(values)
  return x * (1 - x) 

def print_weights_biases(weights, biases):
  print("weights:")
  [print("layer " + str(i) + ":\n" + str(weight)) for i, weight in enumerate(weights)]
  print("end weights")
  print("biases:")
  [print("layer " + str(i) + ":\n" + str(bias)) for i, bias in enumerate(biases)]
  print("end biases")

def cmap_one(i):
    return "#"+hex(((i+1)*2396745)%(256**3))[2:].rjust(6,"0")

def cmap(colors):
    return list(map(cmap_one, colors))

def create_random_weights_biases(feature_dimensions):
  # List of weight matrices. Each matrix_i contains weights for layer_i,
  # and is of shape (dim_(i - 1), dim_i).
  # There are dim_i different nodes,
  # and each of them needs to have dim_(i - 1) weights
  # to connect to all of the previous layers' nodes.
  ws = []
  # List of bias vectors. Each bias vector is of shape (1, dim_i).
  bs = []

  for layer in range(1, len(feature_dimensions)):
    w = np.random.random((feature_dimensions[layer - 1],
                          feature_dimensions[layer]))
    b = np.random.random(feature_dimensions[layer])
    ws.append(w * 2 - 1)
    bs.append(b * 2 - 1)
  return ws, bs

wsandbs = create_random_weights_biases(layer_node_sizes)
weights = wsandbs[0]
biases = wsandbs[1]

def test_result(test_inputs, layers_list, test_weights, test_biases):
  prev = test_inputs
  for i in range(len(layers_list)-1):
    prev = prev.dot(test_weights[i]) + test_biases[i]
    all_node_values.append(prev)
    prev = activation(prev)
    all_activated_values.append(prev)
  return prev

# print_weights_biases(weights, biases)
result = test_result(inputs, layer_node_sizes, weights, biases)
print("result:")
print(result)
print("end result")

def backpropogate():
  global weights, biases, all_node_values, result
  num_tests = result.shape[0]

  cur_layer_derivative = all_activated_values[-1] - answers
  for layer_index in range(len(layer_node_sizes) - 1, 0, -1):

    # activation function's derivative, for everything below
    # returns a vector of the function applied to everything in the original vector
    activation_derivative = get_activation_derivative(all_node_values[layer_index - 1])

    # current layer's difference = loss derivative 
    # because loss function will be x^2 / 2

    # loss for biases
    # sum up for each individual test
    # because biases don't affect derivatives of anything else in this 
    # backpropogation step, can change them now.
    # doing the multiplication on two vectors
    # multiples the elements pairwise
    # which is then juts the derivative
    # also multiply by learning_rate to not overshoot
    # subtract because derivative points to highest change, but we want to minimize loss

    for test_index in range(num_tests):
      biases[layer_index - 1] -= (activation_derivative[test_index] *
                                  cur_layer_derivative[test_index] * 
                                  learning_rate)

    # loss for weights
    # need to multiply the derivative of the activation function,
    # the previous layer's node values, and
    # the derivative of the losses.

    # activation_derivative from above
    # loss derivative is cur_layer_difference
    previous_layer_values = all_activated_values[layer_index - 1]

    # sum the tests together by looking at each test individually

    # create an array of the right size,
    # because weight values affect next_layer_derivative,
    # so can't change until then
    weight_derivatives = np.zeros(weights[layer_index - 1].shape)
    
    for test_index in range(num_tests):
      # get the matrix representing one test,
      # and then flip it (the .T) part
      # so that we can "expand" the matrix later
      prev_layer_flipped = previous_layer_values[test_index:test_index + 1, :].T

      weight_derivatives += (prev_layer_flipped *
                             activation_derivative[test_index] *
                             cur_layer_derivative[test_index] *
                             learning_rate)

    # find next layer's loss
    # mostly identical to weight's derivative

    # weights for this layer
    weight_layer = weights[layer_index - 1]
    # create an array of the right size
    next_layer_derivative = np.zeros((num_tests, layer_node_sizes[layer_index - 1]))
    for test_index in range(num_tests):
      some_of_the_derivative = (activation_derivative[test_index] * 
                                cur_layer_derivative[test_index])
      # this dot is the same as the .T from the last bit,
      # but here, we need to add them all up,
      # and the dot product does that nicely for us.
      next_layer_derivative[test_index] = np.dot(weight_layer, some_of_the_derivative).T

    # actually change weights and current loss derivative
    weights[layer_index - 1] -= weight_derivatives
    cur_layer_derivative = next_layer_derivative

if __name__ == "__main__":
  # run for 100001 backpropogation steps
  for i in range(100001):
    all_activated_values = [inputs]
    all_node_values = []
    result = test_result(inputs, layer_node_sizes, weights, biases)
    
    backpropogate()

# print_weights_biases(weights, biases)
result = test_result(inputs, layer_node_sizes, weights, biases)
print("result:")
print(result)
print("end result")