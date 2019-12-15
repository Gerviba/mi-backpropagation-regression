package hu.gerviba.mi3hf;

import java.util.Arrays;
import java.util.Random;

public final class Neuron {

    private static final Random RANDOM = new Random();

    final int indexInItsLayer;
    final double[] errorDerivatives;
    double input; // x, Zj(L)
    double output; // y
    double deltaY; // MATH: dE/dY
    double deltaX; // MATH: dE/dX
    Neuron[] backward;
    Neuron[] forward;

    // Changes trough the iterations:
    final double[] weights;

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
            weights[i] = (RANDOM.nextDouble() - 0.0) / 1.0;
    }

    public void setOutputManually(double value) {
        output = value;
    }

    @Override
    public String toString() {
        return "Neuron{" +
                "output=" + output +
                ", weights=" + Arrays.toString(weights) +
                "}\n";
    }
}
