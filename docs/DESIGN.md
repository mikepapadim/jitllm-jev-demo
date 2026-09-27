# System One on the GPU, in Java: Jev-compatible decisions with jitLLM + TornadoVM

Status, 2026-09-27: **phase 1 is implemented** (beehive-lab/jitllm#190) and **phase 2 is this repo**. Measured results are in the
[README](../README.md) and the PR. Phase 3 (Kev weights) is not started.

## 1. What Jev is, and why it fits jitLLM

**Jev** (TypeSafe AI, September 2026) is a "System One" model: you send a *state* (text, a list, or
name/value pairs) plus typed *questions*, and it returns typed answers with probabilities instead of
text:

| question | answer |
|---|---|
| `noul` (yes/no) | probability the statement is true, 0..1 |
| `choice` (2+ options) | chosen option, distribution over options, confidence |
| `score` (2-10 ordered levels) | position along the rubric (e.g. 1.05), distribution, confidence |

Wire contract (from the community Java SDK `com.jamilxt:typesafe-ai-java-core`, which also targets
self-hosted servers through `TypeSafeClient.laya(baseUrl, key)`):

```
POST /v1/systemone
{"state": ..., "model": "...", "questions": {
   "is_urgent":  {"type": "noul",   "instructions": "..."},
   "department": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", "technical": "..."}},
   "frustration":{"type": "score",  "instructions": "...", "criteria": ["Calm", "Frustrated", "Very angry"]}}}
->
{"model": "...", "answers": {
   "is_urgent":  {"type": "noul", "noul": 0.93},
   "department": {"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": 0.9},
   "frustration":{"type": "score", "score": 1.05, "legend": {"0": "Calm", ...}, "probabilities": {...}, "confidence": 0.92}},
 "usage": {"input_tokens": 304, "output_tokens": 0}}
GET /v1/models
```

The point of the model class is: **no generation, only scoring**. Questions are evaluated in
parallel, so N questions cost about as much as one. Open-source reproductions show two ways to do
this on an ordinary LLM:

- **Label-token scoring** (jevfire): prefill a shared context once, append each question, read the
  next-token logits of single-token option labels, renormalise over the options. No weights change.
  jevfire reports 8-10x over "generate JSON" for 4-28 fields on vLLM.
- **Trained pointer head** (Kev, Apache-2.0): Qwen 3.5 base + rank-16 LoRA + a small head that
  scores each option's `</opt>` hidden state against the question's `<decide>` hidden state, with a
  fitted softmax temperature for calibration. Speaks `POST /v1/systemone` already.

Everything this needs is something jitLLM already does on the GPU through TornadoVM: prefill,
KV cache with prefix reuse, logits. What it does not do yet is *stop after scoring*.

## 2. What jitLLM gives us today (read at `main` 207cd12c)

| need | jitLLM today |
|---|---|
| prefill a context once, branch many continuations | `SessionRuntime.generateOnGpu(model, startPosition, promptTokens, stopTokens, budget, sampler, onToken)` takes an explicit start position |
| read logits instead of a token | every step calls `Sampler.sampleToken(Logits)`; a capturing sampler sees the full host-side row (on-device argmax is opt-in via `-Djitllm.deviceSample=true`, off by default) |
| serve it | `server/OpenAIServer` (JDK `HttpServer`, own `Json`), `createContext` per route |
| batch many requests with a shared prefix | `LLMEngine` + paged KV + `PrefixCache` (block-level prefix sharing) |
| Qwen 3.5 (Kev's base) | `Qwen35State` exists |

The one subtlety: generation loops feed the state's *seed token* (the last token not yet written to
the KV cache) at the start position before the new prompt (`PromptIngestion`). Branching several
questions from the same prefix position means restoring that seed before each question; otherwise
question 2 would continue from question 1's answer token.

*As built:* loops also disagree on whether the seed is fed at all (the Qwen 3 loops ingest the prompt
from its first token and ignore it), so `DecisionSession` calibrates the convention once per session
on a small probe instead of assuming one. See the PR.

## 3. Phases

**Phase 1 — `POST /v1/systemone` in `jitllm serve` (label-token scoring).**
`DecisionSession` next to `DelegatingSession` (same package, uses `SessionRuntime` directly):
encode the state once through the model's chat template, then per question append
"instructions + lettered options + answer cue", run one step from the prefix position with a
capturing sampler, softmax over the option label tokens. `noul` = P(yes) over {yes, no};
`score` = expected level index over the level labels. Works with every model jitLLM loads.
Acceptance: the unmodified TypeSafe Java SDK, pointed at `jitllm serve`, gets answers for all three
question types; a latency table (per request, per question) vs. generating JSON with the same model.

**Phase 2 — the fancy demo (separate repo, this one).** See section 4.

**Phase 3 — Kev weights.** Merge the LoRA into the Qwen 3.5 base offline, export GGUF, and add
"hidden state at chosen positions" + the pointer head to jitLLM. Turns "prompted LLM that
decides" into "trained System One model", with calibration. Real engine work; only after 1-2.

## 4. The demo: "System One + System Two, one GPU, pure Java"

A live support/moderation desk for a conference (fits a Devoxx stage):

1. A stream of incoming messages (attendee help desk: wifi, lost badge, talk room change, angry
   refund request, security incident...).
2. **System One** (jitLLM `/v1/systemone`, called through the *unmodified* TypeSafe Java SDK)
   triages each message: `is_urgent` (noul), `team` (choice: venue / registration / speakers /
   security / catering), `sentiment` (score: calm → furious), `needs_human` (noul). A web dashboard
   shows the probability bars filling in, and the per-message latency.
3. **System Two** (the same jitLLM model, `/v1/chat/completions`, streaming) drafts a reply only
   for messages System One routes to auto-reply. Urgent + furious goes to a human queue instead.
4. Side panel, the numbers that make the point on stage:
   - ms per decision request and per question, and the shared-prefix token count. *As measured,*
     N questions do **not** yet cost about one: each costs one batched-prefill chunk (~85 ms at 3B).
     One-pass multi-branch prefill is the engine follow-up that would get there;
   - the same triage done the usual way (ask the LLM to emit JSON) and its latency, for contrast;
   - "same weights, same JVM, same GPU; TornadoVM compiled the kernels; no Python, no CUDA C."
5. Swap the TypeSafe `baseUrl` to the hosted API (if a key is present) and nothing else changes —
   the drop-in story.

Honest limits to say out loud: Phase 1 probabilities are relative option preferences of a prompted
chat model, not calibrated confidences (jevfire says the same about its own); calibration is what
Phase 3 (Kev) buys.

## 5. Where it goes

- The server endpoint and `DecisionSession` are general engine features: a PR to `beehive-lab/jitllm`.
- The demo app (dashboard, message stream, SDK usage, benchmark script) lives in this repo.
