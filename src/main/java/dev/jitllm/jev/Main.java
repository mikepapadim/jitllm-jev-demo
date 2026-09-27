package dev.jitllm.jev;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code java -jar jev-demo.jar [dashboard|bench] [--jitllm URL] [--port N] [--pause MS]}
 *
 * <p>Needs a running {@code jitllm serve} with {@code /v1/systemone} (default
 * {@code http://127.0.0.1:8080}).
 */
public final class Main {

    public static void main(String[] args) throws Exception {
        String mode = "dashboard";
        String jitllm = "http://127.0.0.1:8080";
        int port = 9090;
        long pause = 1200;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--jitllm" -> jitllm = args[++i];
                case "--port" -> port = Integer.parseInt(args[++i]);
                case "--pause" -> pause = Long.parseLong(args[++i]);
                case "dashboard", "bench" -> mode = args[i];
                default -> throw new IllegalArgumentException("unknown argument " + args[i]);
            }
        }
        List<String> messages = loadMessages();
        if (mode.equals("bench")) {
            Bench.run(jitllm, messages);
        } else {
            new Dashboard(jitllm, messages, pause).serve(port);
        }
    }

    static List<String> loadMessages() throws Exception {
        try (InputStream in = Main.class.getResourceAsStream("/messages.json")) {
            JsonNode root = new ObjectMapper().readTree(in);
            List<String> out = new ArrayList<>();
            for (JsonNode m : root) {
                out.add(m.path("text").asText());
            }
            return out;
        }
    }
}
