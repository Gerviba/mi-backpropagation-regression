package hu.gerviba.mi3hf;

import java.util.Arrays;

import static hu.gerviba.mi3hf.Main.DATA_LENGTH;
import static hu.gerviba.mi3hf.Main.PRODUCTION;

public final class DataLine {

    final double[] x = new double[81];
    double y;

    public DataLine(double[] data) {
        System.arraycopy(data, 0, this.x, 0, DATA_LENGTH);

        if (data.length == DATA_LENGTH + 1)
            y = data[DATA_LENGTH];
        else if (!PRODUCTION)
            throw new RuntimeException("Y value not presented: " + Arrays.toString(x));
        else
            y = -1;
    }

    @Override
    public String toString() {
        return "DataLine{" +
                "x=" + Arrays.toString(x) +
                ", y=" + y +
                '}';
    }
}
