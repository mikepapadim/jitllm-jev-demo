package dev.jitllm.jev;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * The same triage two ways on the same model: scoring ({@code /v1/systemone}) vs. generating JSON
 * ({@code /v1/chat/completions}). Prints latency, JSON validity and agreement.
 */
final class Bench {

    private static final ObjectMapper JSON = new ObjectMapper();

    private static final String JSON_SYSTEM =
            "You triage conference help-desk messages. Reply with a single JSON object and nothing else,"
                    + " with exactly these fields: \"urgent\" (true if it needs action within 15 minutes),"
                    + " \"team\" (one of: venue, registration, catering, speakers, safety, info),"
                    + " \"mood\" (one of: Happy, Neutral, Annoyed, Angry),"
                    + " \"needs_human\" (true if it needs a person on site),"
                    + " \"safety_risk\" (true if anyone's health or safety is at risk).";

    static void run(String baseUrl, List<String> messages) throws Exception {
        Triage triage = new Triage(baseUrl);
        ChatClient chat = new ChatClient(baseUrl);

        // Warm both paths so compile time is not counted.
        triage.decide(messages.get(0));
        chat.complete(JSON_SYSTEM, messages.get(0), 80);

        List<Double> scoring = new ArrayList<>();
        List<Double> generating = new ArrayList<>();
        int valid = 0;
        int teamAgree = 0;
        int urgentAgree = 0;
        System.out.printf(Locale.ROOT, "%-3s %-9s %-9s %-13s %-13s %s%n",
                "#", "score ms", "gen ms", "team (score)", "team (gen)", "message");
        for (int i = 0; i < messages.size(); i++) {
            String m = messages.get(i);
            Triage.Decision d = triage.decide(m);
            scoring.add(d.millis());

            long t0 = System.nanoTime();
            String text = chat.complete(JSON_SYSTEM, m, 80);
            double genMs = (System.nanoTime() - t0) / 1e6;
            generating.add(genMs);

            String genTeam = "(invalid)";
            JsonNode parsed = parse(text);
            if (parsed != null && parsed.has("team") && parsed.has("urgent")) {
                valid++;
                genTeam = parsed.path("team").asText();
                if (genTeam.equalsIgnoreCase(d.team())) {
                    teamAgree++;
                }
                if (parsed.path("urgent").asBoolean() == (d.urgent() > 0.5)) {
                    urgentAgree++;
                }
            }
            System.out.printf(Locale.ROOT, "%-3d %-9.0f %-9.0f %-13s %-13s %s%n",
                    i + 1, d.millis(), genMs, d.team(), genTeam, abbreviate(m, 60));
        }
        System.out.println();
        System.out.printf(Locale.ROOT, "scoring  (/v1/systemone, 5 typed answers):  median %.0f ms, p90 %.0f ms%n",
                pct(scoring, 50), pct(scoring, 90));
        System.out.printf(Locale.ROOT, "generating JSON (/v1/chat/completions):     median %.0f ms, p90 %.0f ms%n",
                pct(generating, 50), pct(generating, 90));
        System.out.printf(Locale.ROOT, "speed-up (median): %.1fx%n", pct(generating, 50) / pct(scoring, 50));
        System.out.printf(Locale.ROOT, "generated JSON valid: %d/%d; agreement with scoring on team %d/%d, on urgent %d/%d%n",
                valid, messages.size(), teamAgree, valid, urgentAgree, valid);
    }

    private static JsonNode parse(String text) {
        int start = text.indexOf('{');
        int end = text.lastIndexOf('}');
        if (start < 0 || end <= start) {
            return null;
        }
        try {
            return JSON.readTree(text.substring(start, end + 1));
        } catch (Exception e) {
            return null;
        }
    }

    private static double pct(List<Double> xs, int p) {
        List<Double> s = new ArrayList<>(xs);
        Collections.sort(s);
        int idx = (int) Math.ceil(p / 100.0 * s.size()) - 1;
        return s.get(Math.max(0, Math.min(idx, s.size() - 1)));
    }

    private static String abbreviate(String s, int n) {
        return s.length() <= n ? s : s.substring(0, n - 1) + "…";
    }
}
