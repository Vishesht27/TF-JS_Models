const status = document.getElementById("status");
if (status) {
  status.innerText = "Loaded TensorFlow.js - version: " + tf.version.tfjs;
}

import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";

// Input feature pairs (House size, Number of Bedrooms)

const INPUTS = TRAINING_DATA.inputs;

// Current listed house prices in dollars given their features above

// (target output values you want to predict).

const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays in the same way so inputs still match outputs indexes.

tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array of Arrays needs 2D tensor to store.

const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// Output can stay 1 dimensional.

const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

// Function to take a Tensor and normalize values

// with respect to each column of values contained in that Tensor.

function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    // Find the minimum value contained in the Tensor.

    const MIN_VALUES = min || tf.min(tensor, 0);

    // Find the maximum value contained in the Tensor.

    const MAX_VALUES = max || tf.max(tensor, 0);

    // Now subtract the MIN_VALUE from every value in the Tensor

    // And store the results in a new Tensor.

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // Calculate the range size of possible values.

    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    // Calculate the adjusted values divided by the range size as a new Tensor.

    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });

  return result;
}

// Normalize all input feature arrays and then

// dispose of the original non normalized Tensors.

const FEATURE_RESULTS = normalize(INPUTS_TENSOR);

console.log("Normalized Values:");

FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log("Min Values:");

FEATURE_RESULTS.MIN_VALUES.print();

console.log("Max Values:");

FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();
