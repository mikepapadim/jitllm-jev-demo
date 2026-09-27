package dev.jitllm.jev;

import ai.typesafe.TypeSafeClient;
import ai.typesafe.model.ChoiceAnswer;
import ai.typesafe.model.EvaluationRequest;
import ai.typesafe.model.ScoreAnswer;
import ai.typesafe.model.SystemOneResult;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * System One: triage one help-desk message with typed questions, through the TypeSafe Java SDK.
 *
 * <p>The SDK is the unmodified community client for TypeSafe's Jev API. The only line that makes it
 * local is the {@code baseUrl}: point it at {@code https://api.typesafe.ai} (with a real key) and the
 * same code runs against the hosted model.
 */
final class Triage {

    static final Map<String, String> TEAMS = new LinkedHashMap<>();

    static {
        TEAMS.put("venue", "Rooms, AV, projectors, wifi, power, facilities, leaks");
        TEAMS.put("registration", "Tickets, badges, refunds, name changes, swag");
        TEAMS.put("catering", "Food, coffee, dietary needs");
        TEAMS.put("speakers", "Talk schedule, rooms for talks, speaker logistics, slides");
        TEAMS.put("safety", "Medical emergencies, harassment, fire safety, security");
        TEAMS.put("info", "General questions: times, directions, transport");
    }

    static final List<String> MOOD = List.of("Happy", "Neutral", "Annoyed", "Angry");

    /** What System One decided about one message, and how long it took end to end. */
    record Decision(
            double urgent,
            String team,
            Map<String, Double> teamProbabilities,
            double teamConfidence,
            double mood,
            Map<String, Double> moodProbabilities,
            double needsHuman,
            double safetyRisk,
            long inputTokens,
            double millis) {

        /**
         * Routing policy. The thresholds are an application choice, set once for the default model
         * (Llama-3.2-3B): a prompted model's yes/no probabilities lean towards "yes" and are not
         * calibrated, so the bar for "needs a person" is higher than 0.5. A calibrated System One
         * model would let these sit at 0.5.
         */
        String route() {
            if (safetyRisk > 0.7 || (urgent > 0.85 && needsHuman > 0.8)) {
                return "human-now";
            }
            if (needsHuman > 0.75 || mood >= 2.5) {
                return "human-queue";
            }
            return "auto-reply";
        }
    }

    private final TypeSafeClient client;

    Triage(String baseUrl) {
        this.client = TypeSafeClient.builder("local").baseUrl(baseUrl).build();
    }

    Decision decide(String message) {
        EvaluationRequest request =
                EvaluationRequest.of(message)
                        .noul("urgent", "Does this need action within the next 15 minutes?")
                        .choice("team", "Which team should handle this message?", TEAMS)
                        .score("mood", "How does the sender feel?", MOOD)
                        .noul("needs_human",
                                "Does this need a person on site, rather than an automated written reply?")
                        .noul("safety_risk",
                                "Is anyone's health or safety at risk?",
                                "someone could get hurt or is in danger",
                                "no risk to anyone's health or safety")
                        .build();
        long t0 = System.nanoTime();
        SystemOneResult r = client.evaluate(request);
        double ms = (System.nanoTime() - t0) / 1e6;
        ChoiceAnswer team = r.choice("team");
        ScoreAnswer mood = r.score("mood");
        return new Decision(
                r.noul("urgent").noul(),
                team.choice(),
                team.probabilities(),
                team.confidenceOrZero(),
                mood.score(),
                mood.probabilities(),
                r.noul("needs_human").noul(),
                r.noul("safety_risk").noul(),
                r.usage() == null ? 0 : r.usage().inputTokens(),
                ms);
    }
}
