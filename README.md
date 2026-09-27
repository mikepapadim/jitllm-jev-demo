# System One + System Two, one GPU, pure Java

A live conference help desk built on **[jitLLM](https://github.com/beehive-lab/jitllm)** (LLM
inference in Java, GPU kernels JIT-compiled by **TornadoVM**) and the **Jev** "System One" API
contract.

- **System One** triages every incoming message in one call: *urgent?*, *which team?*, *how does the
  sender feel?*, *needs a person?*, *safety risk?*. It returns typed answers with probabilities and
  generates no text. The calls go through the unmodified community Jev Java SDK
  (`com.jamilxt:typesafe-ai-java-core`), pointed at `jitllm serve` instead of `api.typesafe.ai`.
- **System Two** is the same model on the same GPU, generating text. It drafts a reply only for the
  messages System One routes to auto-reply. Emergencies and angry customers go to people.

![dashboard](docs/img/dashboard.png)

## Run it

1. `jitllm serve` with the `/v1/systemone` endpoint (beehive-lab/jitllm#190), on the GPU:

   ```bash
   ./jitllm serve --gpu --model Llama-3.2-3B-Instruct-Q8_0.gguf --ctx-size 4096 \
       --with-prefill-decode --batch-prefill-size 128 --port 8080
   ```

2. The demo (Java 21):

   ```bash
   mvn -q package
   java -jar target/jev-demo.jar dashboard --jitllm http://127.0.0.1:8080   # open http://127.0.0.1:9090
   java -jar target/jev-demo.jar bench     --jitllm http://127.0.0.1:8080   # scoring vs. generating JSON
   ```

The whole System One side is this (`Triage.java`):

```java
TypeSafeClient client = TypeSafeClient.builder("local").baseUrl("http://127.0.0.1:8080").build();
SystemOneResult r = client.evaluate(EvaluationRequest.of(message)
        .noul("urgent", "Does this need action within the next 15 minutes?")
        .choice("team", "Which team should handle this message?", TEAMS)
        .score("mood", "How does the sender feel?", List.of("Happy", "Neutral", "Annoyed", "Angry"))
        .noul("needs_human", "Does this need a person on site, rather than an automated written reply?")
        .noul("safety_risk", "Is anyone's health or safety at risk?", "someone could get hurt", "no risk")
        .build());
```

## What it measured (RTX 4090, CUDA)

System One latency for one message and four questions, warm, median of 5:

| model | latency |
|---|---:|
| Qwen3-0.6B F16 | 80 ms |
| Llama-3.2-1B F16 | 100 ms |
| Llama-3.2-3B Q8_0 | 347 ms |

The same triage (5 typed fields, 24 messages) done two ways on the same model: scoring
(`/v1/systemone`) vs. asking it for JSON (`/v1/chat/completions`). Full tables are in
[`docs/results/`](docs/results/).

| model | scoring | JSON generation | what the JSON path got wrong |
|---|---:|---:|---|
| Llama-3.2-1B F16 | **121 ms** (p90 122) | 227 ms (p90 248) | 2/24 invalid JSON; `team` outside the allowed set ("AV", "room 1", null, ...) |
| Llama-3.2-3B Q8_0 | 432 ms | 411 ms | none invalid; sent the nut-allergy emergency to "info" |

What that says, plainly:

- **Typed answers can't be off-schema.** Scoring only ever picks one of the options it was given,
  with a probability for each. The JSON path produced invalid JSON or values outside the schema on
  the 1B model.
- **On safety, scoring did better on both models.** With 3B, `safety_risk` is 0.98-1.00 on all five
  real safety cases (fainting, being followed, chained fire exit, leak over power strips, nut
  allergy), and all five go to "human now". There is one false positive: a speaker's broken
  projector scores 0.72, which the policy also sends to a person. Everything else is 0.54 or lower.
- **Speed is a draw at 3B today, not 8x.** Each question costs one batched-prefill chunk in jitLLM
  (~85 ms at 3B, whatever its length), so five questions cost five chunks. Scoring every question in
  one prefill pass is the jitLLM change that would turn this into the multi-x win vLLM-based
  implementations report. See the PR's "follow-ups".
- **The probabilities are not calibrated.** A prompted chat model leans "yes", which is why the
  routing thresholds in `Triage.route()` are explicit policy, set once for the model. A trained
  System One head (e.g. Kev) is what makes 0.5 mean 0.5.

## Pieces

| | |
|---|---|
| `src/main/java/dev/jitllm/jev/Triage.java` | System One: the Jev SDK call and the routing policy |
| `src/main/java/dev/jitllm/jev/Dashboard.java`, `src/main/resources/web/index.html` | the live page, server-sent events |
| `src/main/java/dev/jitllm/jev/Bench.java` | scoring vs. generating JSON |
| `src/main/resources/messages.json` | 24 help-desk messages |
| [`docs/DESIGN.md`](docs/DESIGN.md) | what Jev is, how it maps onto jitLLM, the phases |
