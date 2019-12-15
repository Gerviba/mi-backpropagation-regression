package hu.gerviba.mi3hf;

import java.util.Arrays;

import static hu.gerviba.mi3hf.Main.PRODUCTION;

public final class DataLine {

    final double[] x = new double[81];
    double y;

    public DataLine(double[] data) {
        System.arraycopy(data, 0, this.x, 0, 81);

        if (data.length == 82)
            y = data[81];
        else if (!PRODUCTION)
            throw new RuntimeException("Y value not presented: " + Arrays.toString(x));
        else
            y = -1;
    }

    public double[] getX() {
        return x;
    }

    public double getY() {
        return y;
    }

}
