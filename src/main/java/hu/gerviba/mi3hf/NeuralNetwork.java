package hu.gerviba.mi3hf;

import java.util.List;

import static hu.gerviba.mi3hf.Main.PRODUCTION;

/**
 * https://www.youtube.com/watch?v=tIeHLnjs5U8
 * @author Szabó Gergely
 */
public final class NeuralNetwork {

    private final double LEARNING_RATE = 0.02;

    Neuron[] inputNeurons;
    Neuron[][] hiddenNeurons;
    Neuron[] outputNeurons;

    private final int inputNums;
    private final int hiddenLayers;
    private final int nodesPerLayer;

    private DataLine normalMax;
    private DataLine normalMin;
    private double normalFactor;

    public NeuralNetwork(int inputNums, int hiddenLayers, int nodesPerLayer) {
        assert inputNums > 0;
        assert hiddenLayers > 0;
        assert nodesPerLayer > 0;

        this.inputNums = inputNums;
        this.hiddenLayers = hiddenLayers;
        this.nodesPerLayer = nodesPerLayer;

        initNetwork();
        wireNetwork();
    }

    public void train(List<DataLine> data, int withIteration) {

        for (int i = 0; i < withIteration; i++) {
            for (DataLine sample : data) {
                forwardPropagation(sample);
                backPropagation(sample);
            }

            if (!PRODUCTION)
                System.out.println("Iteration " + i + " happened");
        }

        if (!PRODUCTION)
            System.out.println("Learning complete");
    }

    private void forwardPropagation(DataLine sample) {
        for (int i = 0; i < inputNums; i++)
            inputNeurons[i].setOutputManually(normalized(sample.x[i], normalMin.x[i], normalMax.x[i]));

        for (int layerIndex = 0; layerIndex < hiddenLayers; layerIndex++) {
            for (Neuron current : hiddenNeurons[layerIndex]) {
                double Zj = current.bias;

                for (Neuron previous : current.backward)
                    Zj += previous.activation * previous.weights[current.indexInItsLayer];
                current.zSum = Zj;
                current.activation = sigmoidActivationFunction(Zj);

            }
        }

        double Zj = outputNeurons[0].bias;
        for (Neuron previous : outputNeurons[0].backward) {
            Zj += previous.activation * previous.weights[outputNeurons[0].indexInItsLayer];
        }
        outputNeurons[0].zSum = Zj;
        outputNeurons[0].activation = sigmoidActivationFunction(Zj);
    }

    private void backPropagation(DataLine sample) {
        outputNeurons[0].costPerActivationDerivative = 2 * (outputNeurons[0].activation - normalized(sample.y, normalMin.y, normalMax.y));
        double activationPerZDerivative = outputNeurons[0].activation * (1.0 - outputNeurons[0].activation);

        for (Neuron previous : hiddenNeurons[hiddenLayers - 1]) {
            double zPerWeightsDerivative = previous.activation;
            previous.errorDerivatives[outputNeurons[0].indexInItsLayer] =
                    outputNeurons[0].costPerActivationDerivative * activationPerZDerivative * zPerWeightsDerivative;

            previous.weights[outputNeurons[0].indexInItsLayer] -= LEARNING_RATE * previous.errorDerivatives[outputNeurons[0].indexInItsLayer];
        }

        for (int layerIndex = hiddenLayers - 2; layerIndex >= 0; layerIndex--) {
            for (Neuron current : hiddenNeurons[layerIndex]) {

                current.costPerActivationDerivative = 2 * (current.activation - normalized(sample.y, normalMin.y, normalMax.y));
                double currentActivationPerZDerivative = current.activation * (1.0 - current.activation);

                for (Neuron next : current.forward) {
                    double zPerWeightsDerivative = current.activation;
                    current.errorDerivatives[next.indexInItsLayer] =
                            next.costPerActivationDerivative * currentActivationPerZDerivative * zPerWeightsDerivative;

                    current.weights[next.indexInItsLayer] -= LEARNING_RATE * current.errorDerivatives[next.indexInItsLayer];
                }
            }
        }
    }

    private double normalized(double in, double min, double max) {
        return ((in - min) / (max - min)) * normalFactor;
    }

    private double denormalize(double in, double min, double max) {
        return (in / normalFactor) * (max - min) + min;
    }

    private void wireNetwork() {
        for (int inputNodeIndex = 0; inputNodeIndex < inputNums; inputNodeIndex++) {
            inputNeurons[inputNodeIndex].forward = hiddenNeurons[0];
            inputNeurons[inputNodeIndex].bias = 1.0;
        }

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

        outputNeurons[0].backward = hiddenNeurons[hiddenLayers - 1];

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
        double error = 0;
        for (DataLine data : testInput) {
            forwardPropagation(data);
            double result = denormalize(outputNeurons[0].activation, normalMin.y, normalMax.y);

            error += (result - data.y) * (result - data.y);

            if (PRODUCTION)
                System.out.println(result);
            else
                System.out.printf("%4.2f\t%4.2f\tdiff = %4.5f\n", result, data.y, data.y - result);
        }

        error = Math.sqrt((1.0 / Main.INPUT_DATA) * error);
        if (!PRODUCTION)
            System.out.println("RMSE: " + error);
    }

    public static double sigmoidActivationFunction(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    public void setupNormalisation(DataLine max, DataLine min, double factor) {
        this.normalMax = max;
        this.normalMin = min;
        this.normalFactor = factor;

        if (!PRODUCTION) {
            System.out.println(max);
            System.out.println(min);
        }
    }
}
