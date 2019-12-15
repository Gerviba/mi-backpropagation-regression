package hu.gerviba.mi3hf;

import java.util.Random;

public class Neuron {

    private static final Random RANDOM = new Random();

    final int indexInItsLayer;
    final double[] weights;
    final double[] errorDerivatives;
    double output;
    Neuron[] backward;
    Neuron[] forward;

    public Neuron(int indexInItsLayer) {
        this.indexInItsLayer = indexInItsLayer;
        this.weights = new double[0];
        this.errorDerivatives = new double[0];
    }

    public Neuron(int nextLayerNodes, int indexInItsLayer) {
        this.indexInItsLayer = indexInItsLayer;
        this.weights = new double[nextLayerNodes];
        this.errorDerivatives = new double[nextLayerNodes];
        for (int i = 0; i < nextLayerNodes; i++)
            weights[i] = RANDOM.nextDouble();
    }

    public void setOutputManually(double value) {
        output = value;
    }

}
