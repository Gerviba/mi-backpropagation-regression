package hu.gerviba.mi3hf;

import java.util.Arrays;
import java.util.Random;

public final class Neuron {

    static final Random RANDOM = new Random();

    final int indexInItsLayer; // subscript
    final double[] errorDerivatives; // relevant derivatives | dCost/dWeight
    double zSum; // x, Zj(L)
    double activation; // activation(L) = o( Zj(L) )
    double costPerActivationDerivative; // dC0/dActivation(L)
    double bias;
    Neuron[] backward;
    Neuron[] forward;

    final double[] weights;

    {
        bias = (RANDOM.nextDouble() * 2.0 - 1.0) / 1.0;
    }

    public Neuron() {
        this.indexInItsLayer = 0;
        this.weights = new double[0];
        this.errorDerivatives = new double[0];
    }

    public Neuron(int nextLayerNodes, int indexInItsLayer) {
        this.indexInItsLayer = indexInItsLayer;
        this.weights = new double[nextLayerNodes];
        this.errorDerivatives = new double[nextLayerNodes];
        for (int i = 0; i < nextLayerNodes; i++)
            weights[i] = (RANDOM.nextDouble() * 2.0 - 1.0) / 1.0;
    }

    public void setOutputManually(double value) {
        activation = value;
    }

    @Override
    public String toString() {
        return "Neuron{" +
                "activation=" + activation +
                ", weights=" + Arrays.toString(weights) +
                ", bias=" + bias +
                "}\n";
    }
}
