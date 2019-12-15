package hu.gerviba.mi3hf;

import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

public class Main {

    static final int TRAIN_DATA = 17011;
    static final int INPUT_DATA = 4252;
    static final int DATA_LENGTH = 81;
    public static final boolean PRODUCTION = System.getProperty("PROD", "true").equals("true");

    public static void main(String[] args) {
        readInput(System.in);
    }

    static void readInput(InputStream stream) {
        try (Scanner in = new Scanner(stream)) {
            in.useDelimiter("\n");
            List<DataLine> dataset = readData(in, TRAIN_DATA);

            NeuralNetwork neuralNetwork = new NeuralNetwork(DATA_LENGTH, 2, 80);
            normalize(dataset, neuralNetwork, 0.7);

            if (PRODUCTION) {
                double[] yValues = Stream.generate(in::next)
                        .limit(TRAIN_DATA)
                        .filter(line -> !line.isEmpty())
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                for (int i = 0; i < yValues.length; i++) {
                    dataset.get(i).y = yValues[i];
                }
            }

            neuralNetwork.train(dataset, 100);

            List<DataLine> testInput = readData(in, INPUT_DATA);
            neuralNetwork.testResults(testInput);
        }
    }

    private static void normalize(List<DataLine> dataset, NeuralNetwork neuralNetwork, double factor) {
        DataLine max = new DataLine(new double[DATA_LENGTH + 1]);

        double[] minData = new double[DATA_LENGTH + 1];
        final int HUGE_NUMBER = 100000;
        Arrays.fill(minData, HUGE_NUMBER);
        DataLine min = new DataLine(minData);

        for (DataLine data : dataset) {
            for (int i = 0; i < DATA_LENGTH; i++) {
                if (data.x[i] > max.x[i])
                    max.x[i] = data.x[i];
                if (data.x[i] < min.x[i])
                    min.x[i] = data.x[i];
            }

            if (data.y > max.y)
                max.y = data.y;
            if (data.y < min.y)
                min.y = data.y;
        }

        neuralNetwork.setupNormalisation(max, min, factor);
    }

    private static List<DataLine> readData(Scanner in, int inputData) {
        return Stream.generate(in::next)
                .limit(inputData)
                .filter(line -> !line.isEmpty())
                .map(line -> Stream
                        .of(line.split("\t"))
                        .mapToDouble(Double::parseDouble)
                        .toArray()
                )
                .map(DataLine::new)
                .collect(toList());
    }

}
