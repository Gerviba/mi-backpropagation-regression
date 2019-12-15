package hu.gerviba.mi3hf;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Based on: https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/
 * and https://www.youtube.com/watch?v=tIeHLnjs5U8
 * @author Szabó Gergely
 */
public final class NeuralNetwork {

    private final double SCALING_FACTOR = 185.0;
    private final double LEARNING_RATE = 0.8;

    Neuron[] inputNeurons;
    Neuron[][] hiddenNeurons;
    Neuron[] outputNeurons;
    List<Neuron> allNeurons = new LinkedList<>();

    private final int inputNums;
    private final int hiddenLayers;
    private final int nodesPerLayer;

    public NeuralNetwork(int inputNums, int hiddenLayers, int nodesPerLayer) {
        assert inputNums > 0;
        assert hiddenLayers > 0;
        assert nodesPerLayer > 0;

        this.inputNums = inputNums;
        this.hiddenLayers = hiddenLayers;
        this.nodesPerLayer = nodesPerLayer;
    }

    public void train(List<DataLine> data, int withIteration) {
        initNetwork();
        wireNetwork();

        for (int i = 0; i < withIteration; i++) {
            for (DataLine sample : data) {
                forwardPropagation(sample);
                backPropagation(sample);

            }
//            System.out.println("Iteration " + i + " happened");


//            for (int layer = 0; layer < hiddenLayers; layer++) {
//                for (int j = 0; j < nodesPerLayer; j++)
//                    System.out.print(String.format("%3.5f, ", hiddenNeurons[layer][j].weights[0]));
//                System.out.println();
//            }
//            System.out.println("\n!! -> " + outputNeurons[0].output);
//            System.out.println(Arrays.asList(inputNeurons));
//            return;
        }

//        System.out.println("Learning complete");
    }

    private void forwardPropagation(DataLine sample) {
        for (int i = 0; i < inputNums; i++)
            inputNeurons[i].setOutputManually(sample.x[i] / 200.0);

        // MATH: L = layer index
        for (int layerIndex = 0; layerIndex < hiddenLayers; layerIndex++) {
            for (Neuron current : hiddenNeurons[layerIndex]) {
                // MATH: Z_j(L)
                double Zj = 0; // MATH: 0 can be replaced by BIAS_j

                // MATH: A_i(L-1) = output of the previous layer node
                // MATH: sum of [ W_ji(L) * A_i(L-1) ] when i @ edges
                for (Neuron prevous : current.backward)
                    Zj = prevous.output * prevous.weights[current.indexInItsLayer];
                current.input = Zj;
                current.output = sigmoidActivationFunction(Zj);

            }
        }

        // The same thing for the output
        double Zj = 0;
        for (Neuron prevous : outputNeurons[0].backward)
            Zj = prevous.output * prevous.weights[outputNeurons[0].indexInItsLayer];
        outputNeurons[0].input = Zj;
        outputNeurons[0].output = sigmoidActivationFunction(Zj);
    }

    private void backPropagation(DataLine sample) {
        // MATH: E = 1/2 * (Y_output - Y_target)
        // MATH dE/dY_output
        outputNeurons[0].deltaY = outputNeurons[0].output - (sample.y / SCALING_FACTOR);

        // MATH: dE/dX_output = d/dx * f(x) * dE/dY
        double fX = outputNeurons[0].output; //sigmoidActivationFunction(outputNeurons[0].inputNeurons);
        outputNeurons[0].deltaX = fX * (1.0 - fX) * outputNeurons[0].deltaY;

        for (int layerIndex = hiddenLayers - 1; layerIndex >= 0; layerIndex--) {
            for (Neuron current : hiddenNeurons[layerIndex]) {
                // MATH: Null to calc new sum
                current.deltaY = 0;

                // MATH: errorDerivatives[i] = Yi * dE/dXj
                for (Neuron next : current.forward) {
                    current.errorDerivatives[next.indexInItsLayer] = current.output * next.deltaX;

                    // MATH: dE/dY = sum of [ dE/dYi * W_ji(L) ] where i @ forward edges
                    current.deltaY += current.weights[next.indexInItsLayer] * next.deltaY;
                }

                // MATH: dE/dX_output = d/dx * f(x) * dE/dY
                double fXi = current.output; // sigmoidActivationFunction(current.inputNeurons)
                current.deltaX = fXi * (1.0 - fXi) * current.deltaY;
            }
        }

        // Update inputs as well
        for (Neuron current : inputNeurons) {
            for (Neuron next : current.forward) {
                current.errorDerivatives[next.indexInItsLayer] = current.output * next.deltaX;
            }
        }

        // Possibility of wrong function:
        for (Neuron n : allNeurons) {
            for (int edgeIndex = 0; edgeIndex < n.weights.length; edgeIndex++) {
                n.weights[edgeIndex] -= LEARNING_RATE * n.errorDerivatives[edgeIndex];
            }
        }
    }

    private void wireNetwork() {
        // TODO: Init INPUT layer
        for (int inputNodeIndex = 0; inputNodeIndex < inputNums; inputNodeIndex++) {
            inputNeurons[inputNodeIndex].forward = hiddenNeurons[0];
        }

        // TODO: Init HIDDEN layers
        for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++) {
            hiddenNeurons[0][nodeIndex].backward = inputNeurons;
        }

        for (int layerIndex = 0; layerIndex < hiddenLayers - 1; layerIndex++) {
            for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++) {
                hiddenNeurons[layerIndex][nodeIndex].forward = hiddenNeurons[layerIndex + 1];
            }
        }

        for (int layerIndex = 1; layerIndex < hiddenLayers; layerIndex++) {
            for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++) {
                hiddenNeurons[layerIndex][nodeIndex].backward = hiddenNeurons[layerIndex - 1];
            }
        }

        for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++) {
            hiddenNeurons[hiddenLayers - 1][nodeIndex].forward = outputNeurons;
        }

        // TODO: Init OUTPUT layer
        outputNeurons[0].backward = hiddenNeurons[hiddenLayers - 1];

        // TODO: Collect all

        allNeurons.addAll(Arrays.asList(inputNeurons));
        for (int layerIndex = 0; layerIndex < hiddenLayers; layerIndex++) {
            allNeurons.addAll(Arrays.asList(hiddenNeurons[layerIndex]));
        }
        allNeurons.addAll(Arrays.asList(outputNeurons));
    }

    private void initNetwork() {
        inputNeurons = new Neuron[inputNums];
        for (int nodeIndex = 0; nodeIndex < inputNums; nodeIndex++) {
            inputNeurons[nodeIndex] = new Neuron(nodesPerLayer, nodeIndex);
        }

        hiddenNeurons = new Neuron[hiddenLayers][nodesPerLayer];
        for (int layerIndex = 0; layerIndex < hiddenLayers - 1; layerIndex++) {
            for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++)
                hiddenNeurons[layerIndex][nodeIndex] = new Neuron(nodesPerLayer, nodeIndex);
        }

        for (int j = 0; j < nodesPerLayer; j++)
            hiddenNeurons[hiddenLayers - 1][j] = new Neuron(1, j);

        outputNeurons = new Neuron[] { new Neuron() };
    }

    public void testResults(List<DataLine> testInput) {
        for (DataLine data : testInput) {
            forwardPropagation(data);
            double result = SCALING_FACTOR * outputNeurons[0].output;
            System.out.println(result);
//            System.out.printf("%4.2f\t%4.2f\tdiff = %4.5f\n", result, data.y, data.y - result);
        }
    }

    public static double sigmoidActivationFunction(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    public static double square(double x) {
        return x * x;
    }
}
