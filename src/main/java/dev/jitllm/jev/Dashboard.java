package dev.jitllm.jev;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * The stage demo: a live help desk. Messages arrive; System One (the Jev contract, served by jitLLM)
 * triages each in a fraction of a second; System Two (the same model, generating) drafts a reply
 * only for what System One routed to auto-reply. Everything reaches the browser as server-sent
 * events.
 *
 * <p>System One and System Two run one after the other, never concurrently: both use the one model
 * on the one GPU.
 */
final class Dashboard {

    private static final ObjectMapper JSON = new ObjectMapper();

    private static final String REPLY_SYSTEM =
            "You are the help desk of a developer conference. You do NOT know any venue details: no room"
                    + " locations, times, transport lines, prices or procedures. Never state such specifics."
                    + " Write a short, warm reply (at most two sentences): acknowledge the message, give general"
                    + " guidance if it is safe to, and say the help desk team will follow up with the details.";

    private final Triage triage;
    private final ChatClient chat;
    private final List<String> messages;
    private final long pauseMillis;

    Dashboard(String jitllmUrl, List<String> messages, long pauseMillis) {
        this.triage = new Triage(jitllmUrl);
        this.chat = new ChatClient(jitllmUrl);
        this.messages = messages;
        this.pauseMillis = pauseMillis;
    }

    void serve(int port) throws IOException {
        HttpServer http = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        http.createContext("/", this::index);
        http.createContext("/events", this::events);
        http.setExecutor(Executors.newCachedThreadPool());
        http.start();
        System.out.println("dashboard: http://127.0.0.1:" + port + "/");
    }

    private void index(HttpExchange ex) throws IOException {
        try (InputStream in = Dashboard.class.getResourceAsStream("/web/index.html")) {
            byte[] page = in.readAllBytes();
            ex.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
            ex.sendResponseHeaders(200, page.length);
            try (OutputStream out = ex.getResponseBody()) {
                out.write(page);
            }
        }
    }

    /** One run through the message list per browser connection. */
    private void events(HttpExchange ex) throws IOException {
        ex.getResponseHeaders().add("Content-Type", "text/event-stream");
        ex.getResponseHeaders().add("Cache-Control", "no-cache");
        ex.sendResponseHeaders(200, 0);
        try (OutputStream out = ex.getResponseBody()) {
            for (int index = 0; index < messages.size(); index++) {
                final int i = index;
                String text = messages.get(i);
                send(out, "message", Map.of("id", i, "text", text));

                Triage.Decision d = triage.decide(text);
                ObjectNode dn = JSON.valueToTree(d);
                dn.put("id", i);
                dn.put("route", d.route());
                send(out, "decision", dn);

                if (d.route().equals("auto-reply")) {
                    long t0 = System.nanoTime();
                    chat.stream(REPLY_SYSTEM, text, 120, delta -> {
                        try {
                            send(out, "reply", Map.of("id", i, "delta", delta));
                        } catch (IOException closed) {
                            throw new java.io.UncheckedIOException(closed);
                        }
                    });
                    send(out, "reply-done", Map.of("id", i, "millis", (System.nanoTime() - t0) / 1e6));
                }
                Thread.sleep(pauseMillis);
            }
            send(out, "done", Map.of());
        } catch (java.io.UncheckedIOException | IOException closed) {
            // browser went away
        } catch (Exception e) {
            System.err.println("dashboard: " + e);
        }
    }

    private static synchronized void send(OutputStream out, String event, Object data) throws IOException {
        String payload = "event: " + event + "\ndata: " + JSON.writeValueAsString(data) + "\n\n";
        out.write(payload.getBytes(StandardCharsets.UTF_8));
        out.flush();
    }
}
