package hu.gerviba.mi3hf;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.stream.Collectors;

public class BasicTests {

    @TestFactory
    Collection<DynamicTest> basicTests() throws URISyntaxException, IOException {
        return Files.walk(Path.of(ClassLoader.getSystemResource("basic").toURI()))
                .filter(file -> file.toString().endsWith(".dat"))
                .map(file -> DynamicTest.dynamicTest(file.getFileName().toString(), file.toUri(),
                        () -> {
                            System.out.println("Test: " + file.getFileName().toString());
                            Main.readInput(new FileInputStream(file.toFile()));
                        }))
                .collect(Collectors.toList());
    }

}
