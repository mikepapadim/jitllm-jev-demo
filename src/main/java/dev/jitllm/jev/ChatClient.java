package dev.jitllm.jev;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.function.Consumer;

/**
 * System Two: the same jitLLM model, through its OpenAI-compatible {@code /v1/chat/completions}.
 * Used to draft replies (streaming) and, in the benchmark, for the "ask the LLM for JSON" baseline.
 */
final class ChatClient {

    private static final ObjectMapper JSON = new ObjectMapper();

    private final String baseUrl;
    private final HttpClient http =
            HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();

    ChatClient(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    /** Streams the reply token by token to {@code onDelta}; returns the full text. */
    String stream(String system, String user, int maxTokens, Consumer<String> onDelta) throws Exception {
        HttpResponse<java.io.InputStream> resp =
                http.send(request(system, user, maxTokens, true), HttpResponse.BodyHandlers.ofInputStream());
        StringBuilder full = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(resp.body(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (!line.startsWith("data:")) {
                    continue;
                }
                String data = line.substring(5).trim();
                if (data.equals("[DONE]")) {
                    break;
                }
                JsonNode delta = JSON.readTree(data).path("choices").path(0).path("delta").path("content");
                if (!delta.isMissingNode() && !delta.isNull()) {
                    full.append(delta.asText());
                    onDelta.accept(delta.asText());
                }
            }
        }
        return full.toString();
    }

    /** One non-streaming completion. */
    String complete(String system, String user, int maxTokens) throws Exception {
        HttpResponse<String> resp =
                http.send(request(system, user, maxTokens, false), HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200) {
            throw new IllegalStateException("chat completion failed: HTTP " + resp.statusCode() + " " + resp.body());
        }
        return JSON.readTree(resp.body()).path("choices").path(0).path("message").path("content").asText();
    }

    private HttpRequest request(String system, String user, int maxTokens, boolean stream) {
        ObjectNode body = JSON.createObjectNode();
        // No "model": jitllm serve loads one model per process and rejects any other name.
        body.put("max_tokens", maxTokens);
        body.put("temperature", 0.0);
        body.put("stream", stream);
        var messages = body.putArray("messages");
        messages.addObject().put("role", "system").put("content", system);
        messages.addObject().put("role", "user").put("content", user);
        return HttpRequest.newBuilder(URI.create(baseUrl + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofMinutes(2))
                .POST(HttpRequest.BodyPublishers.ofString(body.toString()))
                .build();
    }
}
