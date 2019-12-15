package hu.gerviba.mi3hf;

import java.io.InputStream;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

public class Main {

    private static final int TRAIN_DATA = 17011;
    private static final int INPUT_DATA = 4252;
    public static final boolean PRODUCTION = true;

    public static void main(String[] args) {
        readInput(System.in);
    }

    static void readInput(InputStream stream) {
        try (Scanner in = new Scanner(stream)) {
            in.useDelimiter("\n");
            List<DataLine> dataset = readData(in, TRAIN_DATA);

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

            NeuralNetwork neuralNetwork = new NeuralNetwork(81, 1, 81);
            neuralNetwork.train(dataset, 200);

            List<DataLine> testInput = readData(in, INPUT_DATA);
            neuralNetwork.testResults(testInput);
        }
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
