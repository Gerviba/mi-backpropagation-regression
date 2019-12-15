package hu.gerviba.mi3hf;

import java.util.List;

public final class NeuralNetwork {

    private final double SCALING_FACTOR = 185.0;
    private final double LEARNING_RATE = 0.2;

    Neuron[] input;
    Neuron[][] nodes;
    Neuron[] output;

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

    public void train(List<DataLine> data) {
        initNetwork();
        wireNetwork();

        DataLine sample = data.get(0);
        for (int i = 0; i < inputNums; i++)
            input[i].setOutputManually(sample.x[i]);

        // MATH: L = layer index
        for (int layerIndex = 0; layerIndex < hiddenLayers; layerIndex++) {
            for (Neuron current : nodes[layerIndex]) {
                // MATH: Zj(L)
                double Zj = 0; // MATH: 0 can be replaced by BIASj

                // MATH: Ai(L-1) = output of the previous layer node
                // MATH: sum of [ Wji(L) * Ai(L-1) ] when i @ edges
                for (Neuron prevous : current.backward)
                    Zj = prevous.output * prevous.weights[current.indexInItsLayer];
                current.output = sigmoidActivationFunction(Zj);

                current.errorDerivatives[current.indexInItsLayer] =
                        square(SCALING_FACTOR * output[0].output - sample.y);
            }
        }

        // The same thing for the output
        double Zj = 0;
        for (Neuron prevous : output[0].backward)
            Zj = prevous.output * prevous.weights[output[0].indexInItsLayer];
        output[0].output = sigmoidActivationFunction(Zj);

        // C0 = (Ai(L) - y)^2
        double C0 = (SCALING_FACTOR * output[0].output - sample.y) * (SCALING_FACTOR * output[0].output - sample.y);


    }

    private void wireNetwork() {
        // TODO: Init INPUT layer
        for (int inputNodeIndex = 0; inputNodeIndex < inputNums; inputNodeIndex++) {
            input[inputNodeIndex].forward = nodes[inputNodeIndex];
        }

        // TODO: Init HIDDEN layers
        for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++) {
            nodes[0][nodeIndex].backward = input;
        }

        for (int layerIndex = 0; layerIndex < hiddenLayers - 1; layerIndex++) {
            for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++) {
                nodes[layerIndex][nodeIndex].forward = nodes[layerIndex + 1];
            }
        }

        for (int layerIndex = 1; layerIndex < hiddenLayers; layerIndex++) {
            for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++) {
                nodes[layerIndex][nodeIndex].backward = nodes[layerIndex - 1];
            }
        }

        for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++) {
            nodes[hiddenLayers - 1][nodeIndex].forward = output;
        }

        // TODO: Init OUTPUT layer
        output[0].backward = nodes[hiddenLayers - 1];
    }

    private void initNetwork() {
        input = new Neuron[inputNums];
        for (int nodeIndex = 0; nodeIndex < inputNums; nodeIndex++) {
            input[nodeIndex] = new Neuron(inputNums, nodeIndex);
        }

        nodes = new Neuron[hiddenLayers][nodesPerLayer];
        for (int layerIndex = 0; layerIndex < hiddenLayers - 1; layerIndex++) {
            for (int nodeIndex = 0; nodeIndex < nodesPerLayer; nodeIndex++)
                nodes[layerIndex][nodeIndex] = new Neuron(inputNums, nodeIndex);
        }

        for (int j = 0; j < nodesPerLayer; j++)
            nodes[hiddenLayers - 1][j] = new Neuron(1);

        output = new Neuron[] { new Neuron(0) };
    }


    public void testResults(List<DataLine> testInput) {

    }

    public static double sigmoidActivationFunction(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    public static double square(double x) {
        return x * x;
    }
}
