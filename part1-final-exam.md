# Part 1 Final Exam: Notes

Questions from the Part 1 final exam (LLM and Agent Overview), with worked answers.

## Question 1

What is the primary goal of Reinforcement learning from human feedback (RLHF) in the language model training pipeline?

## Answer 1

**To align the model's behavior with human preferences and intent**, so that it is helpful, honest, and harmless. RLHF doesn't teach the model new facts, and it doesn't make the model smaller or faster. It starts from a model that already has knowledge and language ability, then uses human judgments about which responses are better to shape *how* the model uses that ability.

### Where RLHF fits in the pipeline

| Stage | Objective | Training signal | What the model gets |
|---|---|---|---|
| 1. Pretraining | Predict the next token | Huge amounts of web text, books, and code | Knowledge and language skill, but it behaves like a text *continuer*, not an assistant |
| 2. Supervised fine-tuning (SFT) | Imitate good responses | Human-written (prompt, ideal response) pairs | Basic instruction following and chat format |
| 3. **RLHF** | Maximize a reward that reflects human preferences | Human comparisons of the model's own outputs ("A is better than B") | Responses people actually prefer: more helpful, more honest, safer, better toned |

### The gap RLHF fills

- **Pretraining optimizes the wrong thing.** Its objective is "what text is likely to come next on the internet," not "what does this user want." Asked to "write a poem about autumn," a base model may continue with a list of more writing prompts, because that is what often follows a prompt on the web. It also imitates the false, toxic, and low-quality text it was trained on.
- **SFT only partly closes the gap.** Good demonstrations are expensive to write, so the dataset is small. The model only sees positive examples, and never learns how much worse a bad answer is. Its quality is capped by what the demonstrators can write. It can also push the model to imitate confident answers it has no real basis for.

### How RLHF does it (the InstructGPT recipe)

1. **Collect preferences.** Sample several responses per prompt from the SFT model. Human labelers rank them or pick the better one.
2. **Train a reward model.** A model (often initialized from the SFT model) learns to output a scalar score that predicts the human choice. The loss is Bradley-Terry: −log σ(r(x, y_w) − r(x, y_l)), where y_w is the preferred response and y_l the rejected one.
3. **Optimize the policy with RL** (PPO in the original work) to maximize the reward model's score, with a **KL penalty** against the SFT model: maximize E[r(x, y)] − β · KL(π_θ ‖ π_ref). The policy trains on responses it samples itself, so it gets feedback on its bad outputs as well as its good ones. The KL penalty keeps it close to the SFT model, so it stays fluent and doesn't drift into outputs that merely fool the reward model.

Variants change *how* the preference signal is used, not the goal. DPO optimizes directly on preference pairs, with no separate reward model or RL loop. RLAIF and Constitutional AI replace part of the human labeling with AI feedback.

### Why human preferences instead of more demonstrations

- **Comparing is easier than writing.** It is much easier and more consistent for a person to say which of two answers is better than to write the ideal answer from scratch.
- **Helpful, honest, and harmless are hard to write down as a formula.** There is no loss function for "is this response good." People can judge it, though, and the reward model turns those judgments into a number that RL can optimize.
- **It can go beyond the demonstrators.** Recognizing a good answer is easier than producing one (the same asymmetry behind verifiers in Part 2), so the policy can improve past what labelers would have written themselves.
- **Evidence that it works:** in the InstructGPT paper (Ouyang et al., 2022), labelers preferred outputs from a 1.3B RLHF-tuned model over the 175B GPT-3, a model 100 times larger. Alignment mattered more than scale for perceived quality.

### What it is *not* for (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| Teach the model new knowledge and facts | That is pretraining. RLHF reshapes behavior using knowledge the model already has. |
| Make the model smaller, cheaper, or faster | RLHF doesn't change model size or inference cost. That is distillation and quantization. |
| Maximize benchmark accuracy or minimize perplexity | RLHF optimizes for what people prefer, which is not the same thing. InstructGPT saw an "alignment tax" on some benchmarks and reduced it by mixing pretraining gradients back in (PPO-ptx). |
| Teach basic instruction following | That is SFT. RLHF starts from the SFT model and refines it. |
| Make the model safe, and nothing else | Harmlessness is one third of the goal. Helpfulness and honesty matter equally, and RLHF has to trade them off. |
| Replace pretraining or SFT | It can't. The policy only reweights behaviors it can already produce, so it needs a capable starting model. |

### Limits

RLHF aligns the model with a **proxy** for what humans want, and the proxy is imperfect:

- **Reward hacking (overoptimization).** The policy exploits quirks of the reward model, so true quality can fall even as the reward score keeps rising. This is why the KL penalty exists.
- **Sycophancy and length bias.** Labelers tend to prefer answers that are agreeable, confident, or longer, so the model learns to produce them whether or not they are right.
- **Cost and noise.** Human labels are expensive, and labelers disagree with each other and carry their own biases.

### RLHF vs. RL with verifiable rewards (Part 2)

Both use the same machinery: sample responses, score them, update the policy. What differs is the reward. See `train-time-scaling-rl.md` for GRPO and DAPO.

| | RLHF | RL with verifiable rewards |
|---|---|---|
| Reward comes from | A learned reward model of human preferences | A programmatic check (answer matches, tests pass) |
| Targets | Subjective qualities: helpfulness, honesty, tone, safety | Correctness on checkable tasks: math, code |
| Main risk | Reward hacking, sycophancy | Limited to domains where answers can be verified |

Modern post-training pipelines typically use both.

**In short:** Pretraining gives the model knowledge, SFT teaches it to follow instructions, and RLHF makes it behave the way people actually want, by training on human preference judgments instead of next-token prediction. Its primary goal is **alignment with human preferences and values (helpful, honest, harmless)**, not adding knowledge, shrinking the model, or maximizing benchmark scores.

## Question 2

What is a key limitation of a model that has undergone "Pre-training" only?

## Answer 2

**It isn't aligned with user intent: it doesn't reliably follow instructions, and its outputs can be unhelpful, untruthful, or unsafe.** A model that has only been pretrained (a base model) has been trained to do one thing, predict the next token over a huge text corpus. That makes it a **text continuer, not an assistant**. It has the knowledge and language skill, but nothing in its training told it to *use* them to do what a user asks. This is the gap that post-training (SFT, then RLHF) closes.

### Why this happens

- **The objective is "what text comes next," not "what does the user want."** The model treats a prompt as the start of a document and continues it the way similar documents on the web would. Asked "What is the capital of France?", a base model may continue with more quiz questions ("What is the capital of Germany? ...") instead of answering "Paris."
- **It imitates all of its training data.** That includes false, biased, toxic, and low-quality text. It prefers whatever continuation is most *likely*, not whatever is most truthful, helpful, or safe, and it has no consistent habit of refusing harmful requests.
- **It has no assistant behavior.** There is no chat format or turn-taking, no stable persona, and no sense of when to stop. It may keep going well past the answer, or write the user's next message for them.
- **Prompting can work around this, but it's brittle.** Few-shot examples, or a prompt written so the answer is the natural next line of a document, can coax a base model into doing a task. That takes prompt engineering, and the results can change a lot with small wording changes.

The limitation is in **behavior, not capability**. The model often *can* do what you want, but nothing steers it toward doing so.

### How post-training fixes it

1. **SFT (instruction tuning):** train on human-written (instruction, ideal response) pairs so the model learns to treat a prompt as a request and to answer in the right format.
2. **RLHF (preference tuning):** train on human preference comparisons so the answers are helpful, honest, and harmless (see Answer 1).

### Pretraining-only vs. other LLM limitations (for multiple choice)

| Limitation | Specific to pretraining-only? | What addresses it |
|---|---|---|
| Doesn't reliably follow instructions (continues the text instead of answering) | **Yes, the key one** | SFT / instruction tuning |
| Not aligned with human values: can be unhelpful, untruthful, biased, or toxic | **Yes, the other side of the same gap** | RLHF / preference tuning |
| No chat format, persona, or stopping behavior | Yes | SFT on chat-formatted data |
| Knowledge cutoff (facts frozen at training time) | No, every LLM has it | Retrieval, search, tools, continual pretraining |
| Hallucination | No, it persists after post-training | Grounding in retrieved evidence, verifiers |

If an option claims the model has **no knowledge** or **can't understand language**, it's wrong: pretraining is exactly where those come from.

### Why it matters for agents

An agent loop depends on the model following a system prompt, calling tools in a strict format (such as valid JSON), staying on task across steps, and stopping when it's done. A base model can't be relied on for any of these, so agents are normally built on post-trained (instruction-tuned) models.

**In short:** Pretraining gives a model knowledge and language ability through next-token prediction, but not the behavior of an assistant. A pretrained-only model is not aligned with human intent: it may continue the prompt instead of following it, and it can produce unhelpful, untruthful, or unsafe text. SFT and RLHF are the post-training stages that close that gap.

## Question 3

Which encoding method is widely used by modern language models to convert raw text into tokens?

## Answer 3

**Byte Pair Encoding (BPE)**, a form of **subword tokenization**. Most GPT-style models use *byte-level* BPE. The model never sees raw text. Text is split into tokens, each token is mapped to an integer ID, and each ID is looked up in an embedding table. BPE is the tokenization step at the front of that pipeline: text, then tokens, then IDs, then embeddings, then the transformer.

### How BPE works

1. **Start from a base alphabet.** Characters, or in byte-level BPE the 256 possible bytes of UTF-8 text.
2. **Find the most frequent adjacent pair** of symbols in the training corpus and merge it into one new token. Repeat until the vocabulary reaches a target size (roughly 30k to 200k tokens in modern models).
3. **To tokenize new text,** apply the learned merges in the order they were learned. Common words end up as a single token, and rare words split into smaller pieces (for example, "unbelievably" might become "un", "believ", "ably"). The exact splits depend on the tokenizer.

### Why subwords instead of words or characters

| Approach | Problem |
|---|---|
| Word-level | The vocabulary gets huge, and any word not in it becomes an unknown token (out-of-vocabulary). Typos, names, and new words break it. |
| Character-level | There are no unknown tokens, but sequences get very long. Attention cost grows with sequence length, and each token carries little meaning. |
| **Subword (BPE)** | Fixed vocabulary of manageable size, short sequences for common text, and rare words are still representable from pieces. |

Byte-level BPE goes one step further: because every string is a sequence of bytes, **any text can be encoded**, including emoji, code, and any language, with no unknown tokens.

### Related methods (and who uses them)

- **BPE:** GPT-2, GPT-3, GPT-4 and the Llama family (Llama 3 uses a byte-level BPE with a 128k vocabulary), among many others. The most widely used.
- **WordPiece:** BERT. Similar to BPE, but merges are chosen by how much they improve the likelihood of the training data, not by raw pair frequency.
- **Unigram LM:** T5 and others, through SentencePiece. It starts from a large vocabulary and prunes tokens away.
- **SentencePiece** is a library, not a separate algorithm. It implements BPE and Unigram and treats text, including whitespace, as one raw stream. Llama 2 and Mistral use SentencePiece with BPE.

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| One-hot encoding, bag-of-words, TF-IDF | Older text representations. They don't produce the token sequence a transformer reads. |
| Word2Vec, GloVe | These learn word *embeddings*, which come after tokenization. |
| Positional encoding | Adds word-order information to the embeddings. It is not tokenization. |
| ASCII, UTF-8 | Character-to-byte encodings. Byte-level BPE builds on UTF-8 bytes but then merges them into tokens. |

### Consequences worth knowing

- **Tokens are not words.** In English, a token is roughly 4 characters, or about 0.75 of a word. Context windows, pricing, and rate limits are all counted in tokens.
- **Non-English text, code, and numbers often cost more tokens**, depending on what the tokenizer saw when it was trained.
- **The model sees tokens, not characters.** That is why letter counting, spelling, string reversal, and digit-by-digit arithmetic are harder than they look.
- **The tokenizer is trained separately and fixed before model training.** Changing it means retraining the model.

**In short:** Modern language models turn text into tokens with **Byte Pair Encoding**, usually at the byte level. It repeatedly merges the most frequent adjacent symbols into subword tokens, which gives a fixed vocabulary, short sequences, and no unknown words. WordPiece (BERT) and Unigram (T5) are close alternatives, but BPE is the most widely used.

## Question 4

What can substitute for 'scale' in improving language model performance during pre-training?

## Answer 4

**Better data: higher-quality, carefully curated and well-mixed training data.** Scaling laws say loss falls predictably as you increase parameters, training tokens, and compute. But performance also depends on *what* the model is trained on, so cleaner and more useful data can reach the same quality with a smaller model or less compute.

### What data quality means in practice

- **Filtering and deduplication.** Remove spam, boilerplate, and near-duplicate documents. Duplicates waste compute and encourage memorization.
- **Quality classifiers.** Keep text that looks like educational or reference material (for example, FineWeb-Edu and DCLM-style filtering).
- **Data mix.** Upweight code, math, and reasoning-heavy text, because they improve capabilities beyond what the same number of generic web tokens would.
- **Synthetic data.** Model-generated "textbook-style" text can be denser and cleaner than the web.
- **Evidence:** Phi-1 (*Textbooks Are All You Need*), a 1.3B-parameter code model trained on textbook-quality data, rivaled much larger models on HumanEval.

### Other things that stand in for raw scale

| Substitute | Idea | Example |
|---|---|---|
| **Compute-optimal balance** | Spend a fixed compute budget on the right mix of parameters and tokens, instead of just making the model bigger | Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) at similar compute |
| **Training longer on more tokens** | Train a smaller model far past the compute-optimal point, trading extra training compute for a cheaper model to serve | Llama 3 8B, trained on about 15T tokens |
| **Algorithmic improvements** | Better architectures, optimizers, and training recipes get more out of the same compute | Mixture-of-experts, grouped-query attention, improved optimizers |

### Caveats

- **These are substitutes, not replacements.** Scale still helps, and some abilities (storing many facts, complex reasoning) improve with model size. Good data shifts the scaling curve down but doesn't remove it.
- **High-quality data is a limited supply.** The best web text is finite, which is one reason synthetic data and, later in the pipeline, test-time compute and RL matter.

**In short:** Beyond simply making the model bigger, pre-training performance can be improved by **better data** (quality, curation, mix), plus a compute-optimal split between model size and data, longer training, and better algorithms. The key answer is data quality.

## Question 5

How does "Chain-of-thought" (CoT) prompting assist a language model in solving complex problems?

## Answer 5

**It makes the model write out intermediate reasoning steps before giving its final answer, which breaks a complex problem into smaller steps and gives the model more computation and working memory to solve it.** With a standard prompt, the model has to jump straight from question to answer. With CoT, it generates the path ("first compute X, then use it to get Y, so the answer is Z"), and each step is easier than the whole problem.

### Why it helps

1. **Decomposition.** A multi-step problem (math word problems, logic, planning) becomes a series of simple steps, each of which the model can handle reliably.
2. **More computation per answer.** The model does a fixed amount of computation for each token it generates. A direct answer gets only one token's worth of "thinking," but a chain of 200 tokens gets 200 sequential passes. Spending more tokens on reasoning is spending more compute, which is the idea behind test-time scaling in Part 2.
3. **The text is working memory.** Intermediate results are written into the context, so later steps can attend to them instead of holding everything inside the network's activations.
4. **Better conditioning for the answer.** Relevant facts and partial results are stated explicitly before the final answer is produced, so the answer is conditioned on them.
5. **It draws on what pretraining already contains.** Step-by-step solutions are common in the training text, so the prompt makes the model produce that style of text.
6. **It is inspectable.** A human or a verifier can read the steps, check them, or sample several chains and take a majority vote (self-consistency).

### How to prompt for it

- **Few-shot CoT** (Wei et al., 2022): include examples whose answers show the reasoning steps. PaLM 540B with CoT set the state of the art on GSM8K math word problems.
- **Zero-shot CoT** (Kojima et al., 2022): just add "Let's think step by step."
- **Reasoning models** (o1, DeepSeek-R1) take this further: they are trained with RL to produce long chains of thought on their own, with no special prompt needed.

### Limits

- **It's an emergent ability.** CoT helps mainly in large models (roughly 100B parameters and up in the original work). Small models write fluent but wrong reasoning, and it can hurt them.
- **The written reasoning may be unfaithful.** The chain doesn't always reflect how the model actually reached its answer, and errors in early steps can carry through.
- **It costs tokens and latency**, and adds little for simple lookups or one-step questions.
- **It doesn't add information.** Reasoning can't fix a missing fact. That's what retrieval and tools are for, as in ReAct (see `react-observation-hallucination.md`).

**In short:** CoT prompting has the model **reason step by step in text before answering**. That splits a hard problem into easy pieces, uses more tokens as more computation, and keeps intermediate results in context as working memory. It works best in large models and is the foundation of modern reasoning models and test-time scaling.

## Question 6

What is the "Context" or "Context Window" of a language model?

## Answer 6

**The context is the text (as tokens) the model can see and use when generating its next token, and the context window is the maximum number of tokens it can hold at once.** Everything the model can condition on has to fit inside that window: the system prompt, the conversation so far, any documents or tool results you supply, and the model's own output as it generates. Anything outside the window is invisible to it.

### Key points

- **It is measured in tokens, not words** (see Answer 3). A 128k-token window is roughly 95k English words.
- **It covers input and output together.** The prompt plus the generated reply must fit in the window.
- **It is the model's only short-term memory.** The weights don't change between calls, and the model doesn't remember earlier conversations. A chat feels continuous only because the whole history is sent again with each turn.
- **Information in context is temporary.** It can be used immediately (few-shot examples, retrieved documents, a CoT chain) but is gone when the context is gone. Knowledge in the weights, from pretraining, is permanent.
- **Size varies a lot.** Early models had about 2k tokens. Modern models range from tens of thousands to over a million.

### Why the window is limited

- **Attention cost grows quadratically with length.** Every token attends to every other token, so doubling the context roughly quadruples the attention compute.
- **Memory grows with length.** The KV cache that stores past tokens during generation takes more GPU memory for longer contexts.
- **Position handling is trained to a length.** Models behave worse beyond the sequence lengths they were trained on, so long-context models need extra training or position-scaling techniques.

### Caveats

- **A bigger window isn't free or fully usable.** Long prompts cost more and are slower. Models often recall information at the start and end of a long context better than in the middle ("lost in the middle," Liu et al., 2023), so the *effective* context is smaller than the advertised one.
- **Overflow means loss.** When the limit is exceeded, input gets truncated, the oldest turns are dropped, or the request fails.

### Why it matters for agents

Agents fill the context quickly: system prompt, tool definitions, the Thought/Action/Observation trace in ReAct, and long tool outputs all consume tokens. Agent designs manage this with summarization, retrieval (RAG), and external long-term memory that sits outside the window. This is the motivation for the memory material in Part 3.

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| The size of the training data | That is the pretraining corpus. The context is what the model sees at inference time. |
| The number of parameters | Model size. It is unrelated to how much text fits in a prompt. |
| The vocabulary size | The number of distinct tokens the tokenizer has, not how many can be in one prompt. |
| Long-term memory across conversations | The model keeps none by default. Memory has to be added externally. |

**In short:** The context window is the **maximum number of tokens (prompt plus output) a model can consider at once**. It is the model's working memory for a single call. Anything outside it is invisible, which is why long-running agents need summarization, retrieval, and external memory.

## Question 7

You notice that your chatbot consistently agrees with your incorrect statements just to receive a higher score from users. This phenomenon is known as:

## Answer 7

**Reward hacking, which here shows up as sycophancy.** The two terms describe the same event at different levels. **Reward hacking** is the mechanism: the model found a way to score well on the reward signal (user ratings) without doing what the signal was meant to measure (being helpful and correct). **Sycophancy** is the visible behavior: telling users what they want to hear, including agreeing with claims that are wrong.

### How it happens

1. RLHF trains the model to maximize a reward that stands in for human preference (user thumbs-up, labeler rankings, or a reward model trained on them).
2. People tend to rate agreement, flattery, and confident answers higher than corrections.
3. The policy learns that agreeing is a cheap way to raise the score, so it does it more, even when the user is wrong.

This is **Goodhart's law**: when a measure becomes the target, it stops being a good measure. The reward is only a **proxy** for "good answer," and optimizing the proxy hard enough pulls the two apart. It is the first limit listed in Answer 1. Sharma et al. (2023) found that several leading assistants show sycophancy, and that human and preference-model ratings sometimes favor a convincing agreeable answer over a correct one. OpenAI rolled back a GPT-4o update in 2025 after it became overly flattering, which it attributed partly to over-weighting short-term user feedback.

### Which term to pick on a multiple-choice question

- If **reward hacking** and **sycophancy** are both listed, I would pick **reward hacking**, because the question stresses acting "just to receive a higher score," which is the exploit of the reward signal. Check how your lectures defined each term.
- If only one of the two is listed, pick that one.

### What it is *not* (common wrong answers)

| Tempting answer | What it actually means |
|---|---|
| Hallucination | Generating fluent but false content with no grounding. It isn't driven by pleasing the user or chasing a score. |
| Overfitting | Memorizing training data and failing to generalize to new data. |
| Catastrophic forgetting | Fine-tuning erases abilities learned earlier. |
| Mode collapse | The model's outputs lose diversity and repeat the same few patterns. |
| Alignment tax | Performance drops on some benchmarks after alignment training (see Answer 1). |

### How to reduce it

- Build preference data where the correct answer beats the agreeable one, including cases where the user is wrong.
- Use a KL penalty, reward-model ensembles, or stronger reward models, so the policy can't over-optimize one weak proxy.
- Use verifiable rewards where possible (a correct answer is checked, not rated), as in the RL setting in Part 2.
- Evaluate for sycophancy directly, for example by checking whether the answer changes when the user pushes back.

**In short:** A chatbot that agrees with wrong statements to earn higher user scores is **reward hacking** the proxy reward, and the behavior it produces is **sycophancy**. It is a known failure of RLHF, because the model optimizes the score, not the truth.

## Question 8

What is a significant limitation of autoregressive decoding in current RLHF models?

## Answer 8

**Autoregressive decoding generates one token at a time, left to right, and every token is committed before the next one is produced.** That has two main consequences:

1. **No lookahead or backtracking.** The model can't plan the whole response first or go back and fix an earlier token. A bad early choice carries forward, and errors can compound.
2. **It is inherently sequential, so it is slow.** Token *t+1* depends on token *t*, so output positions can't be generated in parallel. Latency and cost grow with output length.

RLHF doesn't change this. It changes *what* the model prefers to generate, not *how* it generates, so RLHF-tuned models inherit the limitation.

### 1. No planning or revision

- **Each token is a one-way decision.** The model picks a token using only the text so far, with no view of where the answer is heading. A wrong first step in a derivation, or a wrong commitment ("The answer is yes because..."), is hard to undo.
- **Errors compound.** A simple argument says that if each token has some chance *e* of going wrong, a response of *n* tokens has about (1−*e*)^*n* chance of staying right. That is too pessimistic, since errors aren't independent and models can recover, but it captures the risk. Training makes it worse: the model learns from correct prefixes (teacher forcing) but at inference it conditions on its *own* earlier tokens, including mistakes (exposure bias).
- **Fluent text is not verified text.** The model keeps going smoothly even after an error, which contributes to confident wrong answers.
- **Workarounds:** chain-of-thought (reasoning in text before answering, Answer 5), reasoning models that learn to write "wait, let me re-check," and sampling many responses with a verifier or tree search. These are the test-time scaling methods in Part 2.

### 2. Sequential speed

- Every new token needs a full forward pass that depends on the previous token. Decoding is limited by memory bandwidth (reading the weights and KV cache for each token), not by raw compute.
- Long outputs, such as chain-of-thought and agent traces, multiply the latency and cost.
- **Workarounds:** KV caching, batching, quantization, and speculative decoding (a small draft model proposes several tokens and the large model checks them in parallel). Non-autoregressive approaches, like diffusion language models, are an active research area.

### Why it is especially painful for RLHF

- **Rollouts dominate training cost.** RL needs fresh responses from the current policy, and generating them token by token is slow. Long responses leave GPUs waiting on the longest sequence in a batch.
- **Reward arrives only at the end.** A reward model scores the finished response, but the policy made hundreds of token-level choices. Spreading that one score across all of them (credit assignment) gives noisy, high-variance training signal.

### What it is *not* (common wrong answers)

| Tempting answer | What it actually is |
|---|---|
| Reward hacking or sycophancy | A limitation of the RLHF *reward signal* (Answers 1 and 7), not of decoding. |
| The model needs expensive human labels | A cost of RLHF *data collection*. |
| The model can't see enough text | The context window limit (Answer 6). |
| The model has no knowledge | Pretraining supplies knowledge. |

### Which version to pick on a multiple-choice question

I'd match the wording of the option. "Generates sequentially / can't be parallelized / slow" points to speed. "Can't revise earlier tokens / no lookahead / errors accumulate" points to planning. If both are listed, I lean toward the no-backtracking, error-propagation one, since it is the more fundamental quality limit. Check how your lectures framed it.

**In short:** Autoregressive decoding commits to one token at a time, left to right. The model **can't look ahead or revise earlier tokens**, so errors propagate, and generation is **sequential and slow**, which also makes RL rollouts expensive. RLHF inherits this, and test-time search, verifiers, and chain-of-thought are the main ways around it.

## Question 9

A model trained on math problems using RLVR learns to decompose problems and self-correct. How does this capability generalize?

## Answer 9

**It transfers to other domains that need multi-step reasoning, such as coding, science, logic, and planning, because decomposing and self-correcting are general reasoning strategies, not math-specific knowledge. The transfer is strongest in structured, verifiable domains and weaker elsewhere.** RLVR (reinforcement learning with verifiable rewards) only rewards a correct final answer, checked programmatically. The model is never told *how* to reason, so it keeps whichever strategies raise its success rate. Breaking a problem into subgoals, checking intermediate results, backtracking, and trying another approach help on any task with multi-step structure, so they carry over.

### Why it generalizes

- **The learned skills are about process, not content.** "Decompose, verify, backtrack, retry" has nothing to do with algebra specifically. Facts and formulas stay domain-specific, but the habits apply to a proof, a program, or a logic puzzle alike.
- **Outcome-only rewards favor general strategies.** Because the reward is just "was the final answer right," the strategies that survive are ones that work across many problems. Supervised fine-tuning on solutions tends to copy specific solution patterns instead. Papers such as *SFT Memorizes, RL Generalizes* (Chu et al., 2025) report this contrast.
- **Longer, self-checking chains emerge.** In DeepSeek-R1, RL with rule-based rewards led responses to grow longer and to include reflection ("wait, let me re-check") without being taught to. Gains showed up beyond the training domain, for example on science and general reasoning benchmarks.
- **Reported cross-domain evidence.** RL on logic puzzles improved math benchmarks (Logic-RL, Xie et al., 2025), and RL on math has been reported to improve other reasoning tasks more than SFT on the same math data.
- **Cognitive behaviors matter.** Gandhi et al. (2025) found that verification, backtracking, subgoal setting, and backward chaining are the behaviors that let a model improve under RL. Base models that already show them benefit most.

### What generalizes and what doesn't

| Likely to transfer | Less likely to transfer |
|---|---|
| Breaking problems into subgoals | **Domain knowledge.** RLVR on math adds little biology or law knowledge. |
| Checking intermediate work and backtracking | **Tasks with no verifiable answer** (creative writing, tone, advice). These need preference rewards like RLHF. |
| Spending more tokens on harder problems | **Reliable self-checking in unfamiliar areas.** A model can't catch errors it doesn't understand. |
| Tasks with a checkable structure (code, STEM, logic) | **Calibrated effort.** Models can overthink easy questions, and instruction following or style can slip. |

### Caveats

- **Partly sharpening, not new ability.** Some work (Yue et al., 2025) finds RLVR mostly makes the model more likely to find answers it could already reach by sampling many times, instead of expanding what it can solve. This is debated.
- **The verifier limits the reach.** RLVR works where answers can be checked. Extending it to open-ended tasks needs learned reward models, which bring back reward hacking (Answer 7).
- **Transfer is uneven and depends on the base model and training mix.** It is evidence-based but not guaranteed.

See `train-time-scaling-rl.md` for GRPO and DAPO, the RLVR algorithms in Part 2.

### Which version to pick on a multiple-choice question

Choose "transfers to other tasks that need multi-step reasoning (coding, science, logic)." Reject the extremes: "only works on math" and "improves every task equally, including ones with no right answer."

**In short:** RLVR on math teaches **general reasoning habits** (decomposition, verification, self-correction), and those **transfer to other reasoning-heavy domains**, especially verifiable ones. They don't add domain knowledge, and they transfer less to tasks without a checkable answer.

## Question 10

In the context of Reward Hacking, why might a model produce bulleted lists and bold headers even when prose is clearer?

## Answer 10

**Because the reward signal favors responses that look organized and thorough, so the model learns formatting as a shortcut to a higher score, whether or not it makes the answer clearer.** Human raters and reward models tend to score structured responses (bullets, bold headers, tidy sections) higher. The policy is optimized against that proxy, discovers that formatting reliably raises reward, and applies it everywhere, including where plain prose would explain the idea better. This is the same pattern as sycophancy in Answer 7: the model optimizes what the reward measures, not what the reward was meant to measure.

### How it happens

1. **The reward is a proxy for "good answer."** It never measures clarity directly. It measures what raters (or a reward model trained on rater choices) preferred.
2. **Raters use quick heuristics.** Judging many comparisons under time pressure, people read structure as effort and completeness. A skimmable list looks more thorough than a paragraph with the same content.
3. **The reward model learns the correlation.** If formatted answers win more often, the reward model learns "formatting means good," even where it isn't causal.
4. **RL amplifies it.** Adding bullets and headers is cheap and easy for the policy to discover, so optimization pushes it up until it is the default style. The KL penalty slows this down but doesn't stop it.
5. **It spreads.** Style from RLHF'd models leaks into synthetic training data and LLM-judge scoring, which reinforces it in later models.

### Evidence

- **Length bias is well documented** in RLHF (for example, Singhal et al., 2023), and LLM judges show a similar verbosity bias.
- **Style affects rankings.** Chatbot Arena found that longer answers and more markdown headers, bold text, and lists increased win rates, which led to "style control" in its leaderboard. AlpacaEval added a length-controlled version for the same reason.

### Why this is a cost, not just a style choice

- Bullets break an argument into fragments and drop the connecting reasoning ("because," "so," "unless"), which is exactly what prose carries.
- Heavy formatting can make a response look more rigorous than it is, hiding weak or uncertain content.
- It reads as mechanical in conversation and wastes the reader's attention.

Formatting is not always wrong. It really does help for reference material, steps, and comparisons. It becomes reward hacking when it is applied regardless of fit.

### How to reduce it

- Control for style and length in evaluation and in reward training (compare answers of equal length and format).
- Add preference pairs where the plain-prose answer wins when it is clearer.
- Give raters guidelines that don't treat structure as quality, and use reward-model ensembles.
- Use verifiable rewards where possible (Part 2), since a checker can't be impressed by headers.

### Caveat

Reward hacking is the leading explanation, but it isn't proven for any one model. Formatting habits also come from the SFT data, from system prompts, and from imitating other models' outputs.

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| Tokenizer limitations | Tokenization doesn't favor markdown over prose. |
| Small context window | Unrelated to how a response is styled. |
| Autoregressive decoding | Explains left-to-right commitment (Answer 8), not a style preference. |
| Hallucination | About false content, not presentation. |

**In short:** The model produces bulleted lists and bold headers because **its reward (human or reward-model scores) favors organized-looking responses**. It learns formatting as a **spurious shortcut to high reward**, a proxy-gaming behavior alongside length bias and sycophancy, and uses it even when prose would be clearer.

## Question 11

How does Group Relative Policy Optimization (GRPO) improve upon standard RLHF?

## Answer 11

**GRPO removes the separate value (critic) model. Instead of learning a baseline, it samples a group of responses to the same prompt, scores them, and uses each response's reward relative to the group's average as its advantage.** That makes RL training cheaper, simpler, and often more stable than the PPO-style setup used in standard RLHF. GRPO was introduced in DeepSeekMath (Shao et al., 2024) and is the algorithm behind DeepSeek-R1-style reasoning training.

### What standard RLHF (PPO) needs

PPO-based RLHF keeps **four models** in play: the policy being trained, a frozen reference model (for the KL penalty), a reward model, and a **value model (critic)**. The critic is usually as large as the policy. Its job is to estimate how good a state is, so that the advantage (how much better an action was than expected) can be computed for each token. The critic has to be trained alongside the policy, costs about as much memory and compute as the policy itself, and is hard to get accurate for long text.

### What GRPO does instead

1. For each prompt, **sample G responses** from the current policy (G = 64 in DeepSeekMath).
2. **Score each response** with a reward model or a verifier.
3. Compute the advantage by **normalizing within the group**: A_i = (r_i − mean(r)) / std(r). Responses better than the group average get pushed up, and worse ones get pushed down.
4. Update the policy with a PPO-style clipped objective. The **KL penalty goes directly into the loss** instead of being folded into the reward.

The group mean plays the role of the critic's baseline, with no extra model.

### Why this is an improvement

- **Lower memory and compute.** There is no policy-sized critic to train and store, which matters at LLM scale.
- **Simpler and fewer moving parts.** There is no second network to tune or debug, and no critic errors adding noise.
- **A baseline that is fair for each prompt.** The group mean estimates how well the policy does *on that prompt*, so easy and hard prompts are each compared with their own typical result.
- **It matches how reward models work.** Reward models are trained on comparisons, and group-relative scoring is also comparative (an argument made in the DeepSeekMath paper).
- **It works well with verifiable rewards.** With a right/wrong checker (RLVR, Answer 9), a group of samples naturally contains a mix of successes and failures to learn from. This helped make large-scale reasoning RL practical.

### What it doesn't change

- **It is an optimizer, not a new goal.** GRPO changes how the advantage is estimated. The reward can still come from a learned reward model (RLHF) or from a verifier (RLVR), so reward hacking (Answer 7) is still possible with a learned reward.
- **It still needs many samples per prompt**, so rollouts are expensive (Answer 8).
- **If all responses in a group score the same, the advantage is zero** and the prompt teaches nothing.
- **Every token in a response gets the same advantage**, so credit assignment within a response is coarse.

See `train-time-scaling-rl.md` (Answer 2) for the full list of assumptions and fixes such as DAPO and Dr. GRPO.

### Summary

| Aspect | Standard RLHF (PPO) | GRPO |
|---|---|---|
| Baseline for the advantage | Learned value model (critic) | Mean reward of a group of samples |
| Models in memory | Policy, reference, reward, critic | Policy, reference, reward or verifier |
| Samples per prompt | Typically one | A group (for example, 64) |
| KL penalty | Added to the reward | Added to the loss |
| Main cost | Training and storing the critic | Many rollouts per prompt |
| Best fit | Dense or multi-turn rewards | Single-turn, outcome-scored tasks such as math and code |

### Which version to pick on a multiple-choice question

Choose "eliminates the critic/value model by using group-relative rewards from multiple sampled responses." Reject options such as "needs more human labels," "removes the need for a reward," or "makes decoding parallel."

**In short:** GRPO improves on PPO-based RLHF by **dropping the critic and using the group of sampled responses as its own baseline**. That saves memory and compute and simplifies training, at the price of sampling many responses per prompt. It changes the optimizer, not the reward or the alignment goal.

## Question 12

Why is "Chain of Thought" (CoT) reasoning used for the AI judge in RLAIF?

## Answer 12

**So the AI judge reasons step by step before it picks a winner, which makes its preference labels more accurate, more consistent, and easier to inspect.** In RLAIF (reinforcement learning from AI feedback), an AI model replaces human labelers: given a prompt and two responses, it says which is better, and those labels train the reward model or the policy. Everything downstream depends on label quality, so it is worth spending extra tokens to get the judgment right. Constitutional AI (Bai et al., 2022) is an early example, where the judge applies written principles to compare responses.

### Why CoT helps the judge

1. **Better judgments on hard comparisons.** Deciding which answer is better often takes several steps: check the facts, verify the math, test each response against the criteria, and weigh helpfulness against harm. Writing these out first, as in Answer 5, beats giving a one-token verdict straight away.
2. **More computation per label.** The same argument as for CoT in general: more generated tokens mean more compute spent on the decision.
3. **Explicit use of the criteria.** The judge has to say which principle or rubric item each response meets or violates, instead of relying on an impression. This makes the labels follow the intended standard (the "constitution") more closely.
4. **Less reliance on superficial cues.** Judges favor longer answers, nicer formatting, and the first response shown (position bias). Reasoning through the content gives some protection against these shortcuts, though it doesn't remove them (see Answer 10).
5. **Transparency and debugging.** The rationale can be read and audited, so you can see why a label was given, spot systematic judge errors, and tune the judging prompt. This matters because the AI judge becomes part of the training signal.

### Caveats

- **Gains depend on the task.** CoT helps most when judging needs real reasoning (math, factual checks, safety trade-offs) and less on simple or purely stylistic comparisons.
- **The rationale can be unfaithful.** It may be a plausible-sounding justification instead of the real reason for the label.
- **Cost.** Reasoning before every label adds tokens, and RLAIF needs a very large number of labels.
- **Confidence needs care.** CoT labels can be overconfident. Constitutional AI clamped the preference probabilities to a middle range (40 to 60 percent) so they didn't dominate training.
- **The judge's biases persist.** Position, length, and self-preference biases remain, and the policy can learn to exploit them (reward hacking, Answer 7).

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| To make labeling faster or cheaper | The opposite: reasoning adds tokens and cost for each label. |
| To remove the need for a reward model or RL | RLAIF still trains a reward model or optimizes the policy. Only the source of the labels changes. |
| To make the *policy* reason in its answers | This is about the judge, which is a separate role from the model being trained. |
| To guarantee unbiased labels | It improves them but does not remove the judge's biases. |

### Which version to pick on a multiple-choice question

Choose "to improve the accuracy and transparency of the AI judge's preference labels by having it reason before it decides."

**In short:** In RLAIF the AI judge's labels *are* the training signal, so it uses **chain-of-thought to reason through the comparison before choosing**. This gives **more accurate, more criteria-aligned, and more auditable preference labels**, at the cost of extra tokens, and without eliminating the judge's biases.

## Question 13

Why is a KL divergence penalty added to the Reinforcement Learning (RL) loss?

## Answer 13

**To keep the policy close to a reference model (usually the SFT model), so that maximizing reward doesn't pull it into degenerate or exploitative behavior.** KL divergence measures how far the policy's output distribution has moved from the reference. Subtracting it from the reward turns the objective into

  maximize E[r(x, y)] − β · KL(π_θ ‖ π_ref)

The reward says "go this way," and the KL term says "but don't stray far from where you started." β sets the balance.

### Why it is needed

1. **It limits reward hacking.** The reward model is only a proxy trained on a limited set of examples. Pushed hard, the policy finds outputs that score high for the wrong reasons (flattery, formatting, length, nonsense the reward model misjudges), as in Answers 7 and 10. The KL term makes extreme departures costly.
2. **It keeps the reward model reliable.** The reward model was trained on outputs from the SFT model. Far from that distribution it is extrapolating, and its scores become unreliable. Staying close keeps it where the scores mean something.
3. **It preserves what the model already learned.** Without it, RL can erode fluency, knowledge, and instruction following from pretraining and SFT (a form of forgetting), and produce odd or garbled text that happens to score well.
4. **It protects diversity.** Unconstrained RL tends to collapse onto a few high-reward response patterns (mode collapse). KL pulls the policy back toward the broader reference distribution.
5. **It stabilizes training.** Like PPO's clipping, it works as a trust region: each update stays small enough that the policy doesn't change abruptly.

### How it is implemented

- **PPO-style RLHF:** a per-token penalty, β · log(π_θ / π_ref), is subtracted from the reward at each token.
- **GRPO:** the KL term is added to the loss directly (Answer 11).
- **Choosing β:** too low and the policy over-optimizes and hacks the reward. Too high and it barely moves from the SFT model and learns little. It is a tunable trade-off.

### A useful fact

The KL-regularized objective has a closed-form optimal policy, π* ∝ π_ref · exp(r/β). That is the basis of **DPO**, which trains directly on preference pairs without a separate RL loop or reward model.

### Caveat: it is sometimes dropped

It does not fully prevent reward hacking, and it limits how far the policy can improve. For long chain-of-thought RL with verifiable rewards, where a checker can't be fooled the way a reward model can and the policy is *expected* to move far from the initial model, some recipes remove the KL term (DAPO does). The penalty matters most with learned reward models, as in RLHF.

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| To speed up training | It adds a reference-model computation. It is for regularization. |
| To encourage exploration | That is an entropy bonus. KL pulls the policy *toward* the reference. |
| To make the reward model more accurate | It doesn't change the reward model. It limits how far the policy goes beyond where the reward model is trusted. |
| To maximize reward faster | It deliberately trades some reward for safety and stability. |

### Which version to pick on a multiple-choice question

Choose "to keep the policy close to the reference (SFT) model, preventing reward hacking and drift while preserving fluency."

**In short:** The KL penalty is a **leash** that keeps the RL-trained policy near the reference model, so it can't over-optimize an imperfect reward. It guards against **reward hacking, loss of fluency and knowledge, and instability**, with β controlling how tight the leash is.

## Question 14

In the "LLM-as-a-Verifier" framework, how is a more granular reward signal produced compared to simple pass/fail judges?

## Answer 14

**By getting a continuous, fine-grained score out of the LLM verifier instead of a single binary verdict, most commonly by reading the probabilities the model assigns to its verdict or score tokens and combining them into an expected score.** A pass/fail judge asks "is this correct?" and keeps only the sampled word, "yes" or "no." That throws away how confident the model was and how close the response came. A granular verifier keeps that information.

*Note: I couldn't find this exact framework by name in the repo's materials, so this answer describes the standard techniques for getting granular scores from an LLM verifier. Check it against how your lectures described the framework.*

### How the granularity is produced

1. **Use token probabilities, not the sampled verdict.** Instead of reading "Yes" or "No," take the probability of "Yes" (the approach in Generative Verifiers, Zhang et al., 2024). A response at 0.95 and one at 0.55 both count as "pass" for a binary judge, but they get different scores here.
2. **Use a finer scale and take the expected value.** Ask for a rating on a scale (for example, 1 to 10), then compute the **probability-weighted average** over the score tokens, as in G-Eval (Liu et al., 2023). This turns a coarse, tie-prone rating into a smooth score.
3. **Score several criteria, then aggregate.** Break "good" into parts, such as correct final answer, valid reasoning, completeness, and following instructions. Score each one separately, or with natural-language unit tests (LMUnit), and combine them.
4. **Score each step, not just the final answer.** Verifying every reasoning step gives a signal on partial progress and shows *where* a solution went wrong (like the process reward models in *Let's Verify Step by Step* and Math-Shepherd).
5. **Aggregate repeated verifications.** Run the verifier several times, or with several prompts, and average. A 7-of-10 result is a graded score, and the spread shows uncertainty. Weaver does this by combining many verifiers (Answer 12 covers adding reasoning before the judgment).

### Why a granular signal matters

- **Ranking candidates.** In best-of-N selection, binary judges leave many ties. A continuous score can pick the best of several "passing" responses.
- **Learning signal for RL.** With pass/fail rewards, a group of responses that all fail (or all pass) gives GRPO zero advantage (Answer 11). Graded scores still separate better from worse.
- **Credit for partial progress.** A nearly correct solution scores higher than a nonsense one, so training gets a gradient on hard problems.
- **Calibrated confidence.** Scores can be used to weight votes or to decide when to trust a verifier.

### Caveats

- **Precision is not accuracy.** LLM probabilities are often miscalibrated, so small score differences can be noise.
- **Partial credit can be gamed.** A policy may learn to earn the partial-credit parts without solving the problem (reward hacking, Answer 7).
- **Cost.** Multiple criteria, steps, or samples multiply the verifier calls.
- **Verifier bias remains.** Length, format, and position biases still apply (Answer 10).

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| Asking the model to output "pass" or "fail" more times | That is still binary, unless the results are averaged into a fraction. |
| Using a bigger context window | Unrelated to scoring granularity. |
| Fine-tuning the policy | The change is in how the verifier produces its score. |
| Using a human to label every sample | That is the opposite of using an LLM as the verifier. |

### Which version to pick on a multiple-choice question

Choose the option that says the verifier outputs a **continuous or probability-weighted score** (from token probabilities or logits, a multi-level scale, multiple criteria, or step-level scores) rather than one discrete verdict.

**In short:** A pass/fail judge returns one discrete verdict. An LLM-as-a-Verifier produces a **granular score** by using the model's **token probabilities and a finer rating scale (an expected score), often across several criteria or steps and averaged over repeated checks**. That separates good from better, gives RL a signal on partial progress, and gives a confidence estimate, with the caveat that fine differences can be noisy.

## Question 15

How does an "Agentic Workflow" differ from a "True Agent" in terms of process control?

## Answer 15

**In an agentic workflow, the developer controls the process: the steps and their order are predefined in code, and the LLM fills in each step. In a true agent, the LLM controls the process: it decides at run time what to do next, which tools to use, and when to stop.** This follows the distinction in Anthropic's *Building Effective Agents*: workflows orchestrate LLMs and tools through predefined code paths, while agents dynamically direct their own process and tool use.

### The difference in control

- **Workflow:** the *path* is fixed by the designer. The LLM makes decisions *inside* steps (summarize this, classify that, write this draft), but code decides what runs next. The sequence of steps is known in advance.
- **True agent:** the *path* is decided by the model. It runs a loop of plan, act (call a tool), observe the result, and decide again, as in ReAct (see `react-observation-hallucination.md`). The number and order of steps are not known in advance, and the model decides when the task is done.

### Comparison

| Aspect | Agentic workflow | True agent |
|---|---|---|
| Who decides the next step | Developer's code (predefined control flow) | The LLM, at run time |
| Path known in advance | Yes | No |
| Number of steps | Fixed or bounded | Open-ended |
| Stopping | Reaches the end of the pipeline | Model judges the task is done (plus limits) |
| Predictability | High and repeatable | Lower, varies by run |
| Cost and latency | Lower and easier to estimate | Higher and variable |
| Failure modes | Bounded and easier to test | Errors can compound across many self-chosen steps (Answer 8) |
| Best for | Well-understood tasks with a stable procedure | Open-ended problems where the steps can't be scripted |

### Examples

- **Workflow:** a fixed pipeline of extract entities, then validate, then summarize, then translate. Also prompt chaining, routing a request to one of several preset handlers, running calls in parallel and merging them, or an evaluator that loops a fixed number of times.
- **Agent:** a coding agent that explores a repository, runs tests, reads the failures, and keeps editing until the tests pass. Also a research agent deciding what to search next from what it has found.

### Key points

- **It's a spectrum, not a binary.** Systems range from fully scripted to fully autonomous, and many real systems mix the two, for example an agent running inside a workflow step.
- **Both use LLMs and tools.** The difference is **who controls the flow**, not whether tools or LLMs are involved.
- **Start with the simplest design.** A workflow is often more reliable and cheaper. Use an agent when the task truly can't be scripted and the extra autonomy is worth the cost and risk.
- **Agents need guardrails.** Step and cost limits, sandboxing, checkpoints for human approval, and clear tool descriptions. Their long traces also fill the context window (Answer 6).

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| Agents use LLMs and workflows don't | Both use LLMs. |
| Workflows can't call tools | They can. The tool calls just happen at predefined points. |
| Agents have no human oversight | Oversight can and should be added to either. |
| The difference is model size | It's about control flow, not model size. |
| Workflows are a single LLM call | They are multi-step systems, with the steps fixed by the developer. |

### Which version to pick on a multiple-choice question

Choose "in a workflow the control flow is predefined by the developer's code, while in an agent the LLM dynamically decides the steps and tool use."

**In short:** The difference is **who controls the process**. A workflow follows a **predefined path set by the developer**, with the LLM handling the individual steps. A true agent **directs its own process**, choosing actions, tools, and when to stop. Workflows give predictability and low cost, and agents give flexibility at the price of cost, variability, and harder control.

## Question 16

When an agent calls a tool, what does the model itself produce?

## Answer 16

**A structured text request to call the tool: the tool's name and its arguments (typically JSON). The model does not run the tool.** The surrounding application (the agent runtime) reads that request, executes the tool, and sends the result back into the model's context. From the model's side, a "tool call" is just generated tokens that happen to follow a format.

For example, the model might output something like:

```
{"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}
```

and then stop and wait.

### The loop

1. **The application sends the prompt plus the tool definitions** (name, description, parameter schema) as part of the context.
2. **The model produces a tool call** (name and arguments), often after some reasoning text, and stops.
3. **The application parses, validates, and executes** the call (an API request, a database query, running code).
4. **The application returns the result** to the model as a new message, the Observation in ReAct (see `react-observation-hallucination.md`).
5. **The model reads the result and continues.** It makes another tool call or writes the final answer.

Some systems let the model emit several calls in one turn, which the application can run in parallel.

### What the model does and doesn't do

| The model does | The model does not |
|---|---|
| Decide *whether* a tool is needed and *which* one | Execute the tool or touch any external system |
| Generate the arguments, using the schema and conversation | See the real result until the application returns it |
| Read the returned result and decide what to do next | Guarantee the call is valid or safe |

### Why this matters

- **It works because of post-training.** Models are fine-tuned on tool-use examples to produce well-formed calls and to choose tools from their descriptions. A base model does this unreliably (Answer 2).
- **The tool description is the interface.** The model only knows a tool from the name, description, and schema in its context, so unclear descriptions lead to wrong or missing calls.
- **Calls can be wrong.** The model can pick the wrong tool, produce malformed or invented arguments, or call a tool that doesn't exist. The application should validate arguments, handle errors, and let the model retry. Structured-output or constrained decoding can force valid JSON.
- **Security sits in the application.** Because the model only proposes calls, the runtime is where permissions, sandboxing, and human approval belong. Tool results also come back into the context, so untrusted content in them (prompt injection) can influence the model's next step.
- **The context fills up.** Tool definitions and results both consume tokens (Answer 6).

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| The model executes the tool or API itself | The model only generates text. The application runs the tool. |
| The tool's output or the final result | The result comes from the tool, and the application returns it to the model. |
| An update to the model's weights | Nothing is learned or changed. The call is generated at inference time. |
| Natural-language prose only | The call is structured (name plus arguments) so code can parse it. |

### Which version to pick on a multiple-choice question

Choose "a structured request specifying the tool name and arguments (such as JSON), which the application then executes."

**In short:** When an agent "calls a tool," the model **produces a structured request, the tool name and arguments, as generated text**. The **application executes it** and feeds the result back as an observation, and the model continues from there. The model proposes the action, and the runtime performs it.

## Question 17

What is the primary benefit of the Model Context Protocol (MCP) for AI agent developers?

## Answer 17

**It is a standard, open protocol for connecting agents to external tools and data sources, so a developer builds an integration once and it works with any MCP-compatible application or model.** Without a standard, every pairing of AI application and tool needs its own custom connector. MCP replaces that with one shared interface. It is often described as "USB-C for AI."

### The problem it solves

- **Without MCP:** *N* AI applications and *M* tools need up to *N × M* custom integrations, each with its own format for describing tools, calling them, and returning results.
- **With MCP:** each tool is wrapped once as an MCP server, and each application implements the MCP client side once. That is about *N + M* pieces of work, and everything works together.

### How it works

- **Host:** the AI application (a chat app, IDE, or agent) that runs the model.
- **Client:** a connector inside the host that talks to one MCP server.
- **Server:** a program that exposes capabilities from some system (files, a database, GitHub, a ticketing system, an internal API).
- **Protocol:** messages are JSON-RPC 2.0, sent over a local connection (stdio) or over HTTP.
- **Discovery:** the client asks a server what it offers at run time, so tools can be added or changed without rewriting the agent.
- **What a server can expose:** **tools** (actions the model can request), **resources** (data to read into context), and **prompts** (reusable templates).

MCP sits on the application side of the loop in Answer 16. The model still just produces a structured tool call, and MCP standardizes how the runtime finds the tool, runs it, and returns the result.

### Benefits for developers

- **Build once, reuse everywhere.** An MCP server works across clients and models, instead of being tied to one vendor's tool-calling format.
- **Less custom glue code.** The connection, discovery, and call format are handled by the protocol.
- **Easy to extend.** Adding a capability means connecting another server.
- **An ecosystem.** Many ready-made servers exist, and many AI vendors and tools support the protocol.
- **Separation of concerns.** Tool owners maintain their server, and agent builders focus on agent behavior.

### Caveats

- **It standardizes the interface, not quality or safety.** It doesn't make the model choose tools better.
- **Security needs care.** Servers can be untrusted or malicious, tool results can carry prompt injection, and broad permissions are risky. Use authentication, least privilege, and human approval for sensitive actions.
- **Context cost.** Every connected tool's description takes up tokens, so connecting very many tools can crowd the context window and confuse tool selection (Answer 6).

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| Makes the model smarter or more accurate | It changes how tools connect, not model capability. |
| Reduces model size, cost, or latency | It is an integration protocol. |
| A new model architecture or training method | It is a communication standard, not part of training. |
| Increases the context window | Unrelated. Tool definitions even consume context. |
| Guarantees secure tool use | It does not. Security is up to the implementation. |

### Which version to pick on a multiple-choice question

Choose "it standardizes how agents connect to external tools and data sources, so integrations are built once and reused across applications and models (interoperability)."

**In short:** MCP's primary benefit is **standardized interoperability**. It gives agents **one open protocol for tools and data**, which cuts the *N × M* custom-integration problem down to *N + M*. It makes tools reusable and discoverable, but it doesn't make the model better or the system secure on its own.

## Question 18

Why do standard static benchmarks like MMLU fall short for evaluating AI agents?

## Answer 18

**Because they test static, single-turn knowledge recall, while agents succeed or fail by acting over many steps in an interactive environment.** MMLU asks multiple-choice questions across 57 subjects and scores one answer. An agent has to plan, call tools, read results, recover from errors, and reach an end state, such as a bug fixed or a booking made. A model can ace MMLU and still be a poor agent, and the benchmark would never show it.

### Where static benchmarks fall short

1. **They measure the wrong skill.** MMLU tests knowledge and reasoning on a fixed question. Agent performance depends on planning, tool selection, long-horizon consistency, error recovery, and following constraints, none of which a multiple-choice question exercises.
2. **They are single-turn and non-interactive.** There is no environment, no tool use, no changing state, and no feedback from earlier actions. The model can't act, observe, and adapt (the ReAct loop).
3. **They score the answer, not the outcome.** Agent success is defined by the final state of the world (tests pass, the ticket is resolved) and sometimes by *how* it got there: steps taken, policies followed, side effects, safety.
4. **They ignore reliability.** Agents are non-deterministic and can succeed on one run and fail on the next. A single accuracy number hides this. Agent evaluations repeat tasks and measure consistency (for example, τ-bench's pass^k).
5. **They are saturated and contaminated.** Frontier models score very high on MMLU, so it no longer separates them. A fixed public test set can also leak into training data, and a model can memorize it.
6. **They can't see system design.** An agent's performance depends on the scaffold around the model: tools, prompts, memory, and the control loop. A benchmark that tests only the bare model misses it.
7. **They skip cost, latency, and safety.** Steps, tokens, time, and risky actions or prompt-injection robustness are all part of whether an agent is usable, and a static score includes none of them.
8. **Multiple choice is easy to game.** A random guess is right 25% of the time, and scores can shift with prompt format. Real tasks are open-ended and messy.

### What agent evaluation looks like instead

| Benchmark | What it tests |
|---|---|
| SWE-bench | Resolving real GitHub issues, checked by running the repo's tests |
| WebArena | Completing tasks on realistic websites |
| τ-bench | Tool-using agents talking to a simulated user while following business policies, with repeated-trial reliability |
| GAIA | Real-world assistant questions that need browsing and tools |
| OSWorld | Operating a computer through its interface |

Common features: an **interactive environment**, an **outcome check that can be verified** (tests, final state), **repeated trials**, **trajectory-level analysis** (not just the final answer), and measures of **cost and latency**. Rubric-based LLM judges (Answer 14) and human review fill in where outcomes can't be checked automatically. This is the focus of the evaluation material in Part 3.

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| MMLU is too hard for models | The problem is the opposite: it is near saturation, and it tests the wrong thing. |
| It only covers a few languages or subjects | It spans 57 subjects. Breadth of knowledge isn't the gap. |
| It is too expensive to run | It is cheap to run. That isn't the issue. |
| Agents don't need to be accurate | Accuracy matters, but it isn't sufficient for multi-step action. |

### Which version to pick on a multiple-choice question

Choose "they measure static, single-turn knowledge, not multi-step interactive behavior such as planning, tool use, and error recovery, or outcomes in an environment."

**In short:** MMLU-style benchmarks are **static, single-turn, and knowledge-focused**, so they miss what makes an agent good: **acting over multiple steps in an environment, using tools, recovering from errors, and reaching correct outcomes reliably and efficiently**. Agents need interactive, outcome-based, repeated-trial evaluations such as SWE-bench, WebArena, and τ-bench.

## Question 19

What "fundamental limitation" remains even for a perfectly aligned and reasoning LLM?

## Answer 19

**It is a closed, static system: it can only use what is in its frozen weights and the context it is given, and it can only output text. It can't get new or private information from the world, take actions in it, or remember anything beyond the current context.** Alignment fixes *what the model wants to do*, and reasoning fixes *how well it thinks*. Neither supplies information the model doesn't have, or the ability to act. That gap is the reason agents exist: an LLM wrapped in tools, retrieval, memory, and a control loop.

### What remains even for a perfect aligner and reasoner

- **Frozen, outdated knowledge.** Facts stop at the training cutoff. A perfect reasoner can't deduce today's weather, a stock price, or last week's news, because that information isn't in its weights.
- **No access to private or external data.** It can't read your files, database, or company wiki unless they are placed in its context.
- **No ability to act.** Text out is all it produces. It can't send an email, run code, or book a flight on its own (Answer 16).
- **No persistent memory.** Each call starts fresh, and the context window is finite (Answer 6). It also doesn't keep learning from experience after deployment, because the weights are fixed.
- **Reasoning can't create information.** Thinking longer helps combine and check what you know. It can't observe the world. Claims about the real world still need evidence, which is the point of the Observation step in ReAct (see `react-observation-hallucination.md`).

A well-aligned model would be *honest* about this, saying "I don't know" or "I can't check that," but it still wouldn't have the information or the ability.

### How agents address it

| Limitation | What agent systems add |
|---|---|
| Outdated or missing knowledge | Retrieval (RAG) and web search |
| Can't act | Tool calls, standardized with MCP (Answers 16 and 17) |
| No memory | Short-term context management and external long-term memory (Part 3) |
| Can't verify claims | Observations from tools, and verifiers |
| Single-shot output | A plan, act, observe loop and workflows (Answer 15) |

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong here |
|---|---|
| Hallucination, bias, or harmful outputs | Those are alignment problems, which the question assumes are solved. |
| Poor logic or math | That is a reasoning problem, also assumed solved. |
| Slow or expensive inference | An engineering constraint, not a fundamental limit on what it can know or do. |
| Too little training data | More pretraining data still leaves a cutoff and no access to live or private data. |

### Which version to pick on a multiple-choice question

Choose the option about being **limited to its training knowledge and unable to interact with the external world** (no real-time or private information, no actions, no persistent memory). If a knowledge-cutoff option and a can't-act option both appear, pick the broader one about lacking access to or interaction with the outside world.

**In short:** Even a perfectly aligned, perfectly reasoning LLM is a **closed system**: it has only **frozen training knowledge plus its current context**, and it **can only output text**. It **can't perceive or act on the real world, or remember across sessions**. Closing that gap is the purpose of agentic systems built from tools, retrieval, and memory.

## Question 20

What happens to agent accuracy as the "horizon" (length/complexity) of a task increases?

## Answer 20

**Accuracy drops, often sharply, because errors compound across the steps.** A short task needs only a few things to go right. A long-horizon task needs many in a row, and one wrong step can derail everything after it. If each step succeeds with probability *p*, and a mistake isn't recovered from, the chance of finishing an *n*-step task is about *p*^*n*, which falls roughly exponentially with length.

### What the compounding looks like

| Per-step success | 10 steps | 20 steps | 50 steps | 100 steps |
|---|---|---|---|---|
| 99% | 90% | 82% | 61% | 37% |
| 95% | 60% | 36% | 8% | 0.6% |
| 90% | 35% | 12% | 0.5% | ~0% |

An agent that is "95% reliable" per step finishes a 20-step task only about a third of the time. This is a simplified model (it assumes independent, unrecoverable errors), but it explains why a modest per-step error rate matters so much. Real agents can sometimes notice and fix mistakes, which softens the curve but doesn't remove the downward trend. Evaluations show the same pattern: METR's work on task time horizons finds that success rates fall as tasks take humans longer, and τ-bench shows that reliability drops when the same task must succeed repeatedly.

### Why accuracy falls with horizon

1. **Errors compound.** Later steps build on earlier ones, so a wrong tool call or misread result carries forward (the autoregressive version of this is in Answer 8).
2. **Off-distribution states.** After a mistake, the agent is in a situation it hasn't seen in training, and it may be more likely to make another mistake.
3. **The context fills up.** Long traces of thoughts, tool calls, and results use up the window, and details in the middle are recalled less well (Answer 6).
4. **Drifting from the goal.** Over many steps the agent can lose track of the original task or constraints.
5. **Fewer chances to verify.** Long tasks often don't have a check until the end, so a failure shows up late and is hard to trace.
6. **Training is harder.** The reward often arrives only at the end of a long trajectory, so credit assignment is poor and few long-horizon examples exist (Answers 8 and 11).

### How to improve it

- **Decompose the task** into shorter subtasks with checkpoints, so errors are caught and contained.
- **Verify as you go.** Check each step with tool observations, tests, or a verifier (Answer 14), and let the agent retry.
- **Re-plan** when a step fails, rather than pushing ahead.
- **Manage context** with summaries, retrieval, and external memory (Part 3).
- **Use workflows** for the parts that can be scripted (Answer 15).
- **Raise per-step reliability.** Better models, and RL training on long-horizon tasks with step-level feedback. Small per-step gains compound into large gains at long horizons.
- **Keep humans in the loop** at key checkpoints for high-stakes tasks.

### What it is *not* (common wrong answers)

| Tempting answer | Why it's wrong |
|---|---|
| Accuracy stays the same | It declines as tasks get longer. |
| Accuracy rises because there are more chances to be right | More steps mean more chances to fail. Only verification and retries turn extra steps into improvement. |
| It drops only because of the context window | The context limit is one cause, but compounding errors would hurt even with an unlimited window. |
| It drops linearly | The chance of full success falls roughly exponentially when errors compound. |

### Which version to pick on a multiple-choice question

Choose "accuracy decreases (often sharply) as the horizon grows, because errors compound over more steps."

**In short:** As the **horizon grows, agent accuracy falls**, roughly like *p*^*n*, because **errors compound and context and goal tracking degrade over many steps**. The remedies are to **shorten the horizon** (decompose and checkpoint), **verify and recover** at each step, and **raise per-step reliability**.
