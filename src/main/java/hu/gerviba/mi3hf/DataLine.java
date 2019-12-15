package hu.gerviba.mi3hf;

public final class DataLine {

    final double[] x = new double[81];
    final double y;

    public DataLine(double[] data) {
        System.arraycopy(data, 0, this.x, 0, 81);
        y = data[81];
    }

    public double[] getX() {
        return x;
    }

    public double getY() {
        return y;
    }

}
