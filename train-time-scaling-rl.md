# Train-Time Scaling / Scaling RL: STaR Rationalization and GRPO's Group Baseline

Questions for Part 2, §3.2, on *STaR: Bootstrapping Reasoning With Reasoning* (Zelikman et al., 2022) and GRPO from *DeepSeekMath* (Shao et al., 2024).

## Question 1

STaR's key trick, rationalization, hands the model the correct answer and has it reason backward to justify that answer. It then fine-tunes on the resulting rationale with the hint removed, as if the model had solved the problem cold. That is a fundamentally different task from searching for an answer you don't already have. Does reasoning backward from a given answer build the same kind of reasoning skill as reasoning forward would, or is the model learning something subtly different?

## Answer 1

Partly the same, and subtly different in the places that matter most. Rationalization teaches the model **what a correct solution looks like**. It doesn't teach **how to find one**. For steps that follow directly from the previous step, the two are the same thing. They split apart at decision points (which approach to take, when to give up on a path) and at "leap" points, where the hinted model can simply move toward the answer it already knows.

### Why the two tasks are different

- **Forward reasoning is search under uncertainty.** The model has to choose an approach, follow it, notice dead ends, and check whether the result makes sense. Most of the difficulty is in the choices, not in carrying out the steps.
- **Backward reasoning is filling in a path to a known endpoint.** Once the destination is fixed, the hard choice is mostly made for you: "the answer is 42, so it must be the approach that gives 42." What remains is building a path that ends there.
- **The training data hides this difference.** Once the hint is removed, a rationalized trace reads exactly like a forward solution. But it was sampled from p(rationale | question, **answer**) and is being trained on as if it came from p(rationale | question). The text looks the same, but it was produced by a different process.

### What the model learns that is subtly different

1. **Clean derivations with no search in them.** Rationalized traces don't contain wrong turns, backtracking, or "wait, that doesn't work," because the model never needed any. Training on them teaches the model to write confident, straight-line solutions. It doesn't teach it to recover from a bad start, which is the skill it needs most on problems it couldn't already solve. (Compare the long, self-correcting chains of thought that come out of on-policy RL in o1/R1-style training.)
2. **Leaps that are only justified by the hint.** The answer *causes* the rationale. The hinted model can make an unjustified jump at the crucial step, like "therefore x = 7," and still reach the right answer. STaR's filter checks only the **final answer**, so these traces pass. After fine-tuning, the model has learned that a confident leap at the hard step is fine. At test time, with no hint, it makes the same kind of leap and lands on a wrong answer. This is post-hoc rationalization, the same failure that makes chains of thought unfaithful (Turpin et al., 2023), now baked into the training data.
3. **The hardest skill gets the weakest training signal.** The bottleneck in a hard problem is usually the key insight or the choice of approach. Given the answer, that insight is often obvious or can be skipped. So rationalization trains the easy, mechanical steps well and the critical choice poorly, which is the opposite of what the failed problems needed.
4. **Shortcuts tied to the answer.** On multiple choice (CommonsenseQA, STaR's main benchmark), reasoning backward can become "explain why this option's wording fits." That's pattern matching on the answer, and it won't transfer to a setting where the model has to come up with the answer itself.

### Why it still works anyway

- **A valid derivation is valid no matter how it was found.** If a rationalized trace is actually correct step by step, it really is a forward solution. Humans learn this way all the time: every textbook worked example was written by someone who already knew the answer.
- **It gives the model a learning signal where it otherwise has none.** Without rationalization, STaR only learns from problems it already solves, and improvement levels off. Rationalization gives a gradient on problems beyond the model's current ability, which is the point of the trick. The paper reports that it improves results and speeds up learning, for example on multi-digit arithmetic.

### A cleaner way to see it: biased posterior sampling

STaR without rationalization approximates policy-gradient RL with a 0/1 reward. That is on-policy training: the model learns from rationales it actually produced. Rationalization replaces this with samples from a different distribution, the **hinted** model, and adds no correction for the mismatch.

There is a principled version of this idea. Treating the rationale as a latent variable, training on samples from the true posterior p(r | x, y) ∝ p(r | x) · p(y | x, r) is EM on the marginal likelihood of the answer. The true posterior only gives weight to rationales the **unhinted** model could plausibly have produced. **The hinted prompt is not that posterior.** It happily produces rationales that only make sense because of the hint. Those are exactly the leaps and shortcuts above, and the unhinted model would almost never produce them. The gap between "what the model writes when told the answer" and "the model's own posterior over rationales" is where the subtle difference comes from. Later work makes this criticism directly and replaces rationalization with MCMC-style posterior sampling (TRICE; Phan, Hoffman et al., 2023). Scaling-RL pipelines like DeepSeek-R1 also mostly avoid answer-conditioned generation and rely on on-policy rollouts plus many samples.

### How you would test it

- **Likelihood gap:** score rationalized traces under the *unhinted* model. If they get much lower likelihood than forward traces, they are off-distribution, and the model is learning to imitate something it can't produce on its own.
- **Step-level validity:** run a process reward model or a human check over rationalized and forward traces that reached the same correct answer. My prediction: rationalized traces have more invalid or skipped steps, concentrated at the key step.
- **Error profile after training:** compare wrong answers from models trained with and without rationalization. My prediction: the rationalization-trained model is wrong more confidently, with smooth derivations that make an unjustified leap, rather than with visible confusion.

### Making it closer to forward reasoning

- Filter rationalizations by checking them **step by step**, not just by final answer.
- Drop traces whose likelihood under the unhinted model is very low, or reweight them (an importance-weighting or MCMC approach).
- Give **weaker hints**, such as the first step or the approach instead of the final answer, so the model still has to do part of the search.
- Use rationalization only to bootstrap, then switch to on-policy RL so the model learns from its own exploration.

### Summary

| Aspect | Forward reasoning (on-policy) | Backward rationalization |
|---|---|---|
| Sampling distribution | p(r \| x), the model's own | p(r \| x, y), hinted and off-policy |
| Hard choice (approach / insight) | Must be found | Mostly given away by the answer |
| Search behavior in traces | Backtracking, checking, recovery | Clean straight lines, no recovery |
| Filter checks | Final answer | Final answer only, so invalid leaps pass |
| What gets learned | How to find a solution | What a solution looks like |
| Main benefit | Faithful to what the model can do | Learning signal on problems it can't yet solve |
| Main risk | Gets stuck at the model's current ability | Confident unjustified leaps, unfaithful CoT |

**In short:** Rationalization teaches the model the *content* of good reasoning without the *process* that finds it. Where the content is correct step by step, that transfers: it's learning from worked examples. Where the hint did the work, the model learns to write convincing paths toward answers it no longer has, and it brings that habit to test time. That makes rationalization a useful way to bootstrap, but a biased substitute for real forward search. It's also why later train-time scaling work moved toward on-policy RL and proper posterior sampling.

## Question 2

GRPO removes PPO's separate critic model entirely. Instead it estimates a baseline by sampling 64 answers to the same question and normalizing each score against that group's own mean and spread. That substitute only works as well as the critic it replaces if those 64 samples actually span a meaningful range of good and bad answers. What is GRPO implicitly assuming about that group of 64 sampled answers for the group-based baseline to be a fair substitute for a learned critic, and when might that assumption not hold?

## Answer 2

For each question GRPO samples G = 64 answers from the current policy, scores them, and gives answer *i* the advantage

  A_i = (r_i − mean(r)) / std(r)

applied to every token of that answer. The group mean stands in for the critic's value estimate V(question): "how well does the policy usually do here?" Dividing by std rescales each question's advantages to unit spread. For that to work, GRPO is implicitly making five assumptions about the group.

### The implicit assumptions

1. **The group has reward variance: some answers succeed and some fail.** The whole signal is *relative*. If all 64 are right or all 64 are wrong, every advantage is 0 and the question contributes no gradient. A critic has the same problem in this case (no reward differences means nothing to learn from), but GRPO still pays for 64 rollouts to find out.
2. **64 on-policy samples estimate the question's value well.** The mean is a Monte Carlo estimate of the pass rate under the *current* policy. With binary rewards its standard error is √(p(1−p)/64), about ±0.06 at p = 0.5, which is fine. This assumes the samples are independent draws from the policy, not near-duplicates.
3. **One scalar per answer is enough credit assignment.** A critic can give per-token values: "this step made success more likely." GRPO gives every token in an answer the same advantage and relies on comparisons *across* the 64 answers to sort out, on average, which steps matter. That assumes the outcome reflects the whole answer, and that good and bad steps are spread across answers in a way that averages out.
4. **Within-group differences in reward reflect real differences in quality.** z-scoring takes whatever spread exists and scales it to unit size. That's harmless with clean 0/1 verifiable rewards. With a noisy reward model, a group whose scores differ only by noise gets advantages just as large as a group with real right and wrong answers.
5. **Every question should get equal weight, whatever its difficulty.** Dividing by std means a question at 63/64 correct and one at 32/64 produce advantages of the same overall scale. With binary rewards and pass rate p, a correct answer gets +√((1−p)/p) and a wrong one gets −√(p/(1−p)). At p = 1/64 the single success gets about **+7.9** and each failure about −0.13. GRPO is betting that a rare success is real signal worth amplifying.

### Why this is often a *fine* substitute

The bar isn't high. A learned critic for LLM reasoning has to predict, from a partial chain of thought, the chance of eventually getting the right answer. That's about as hard as the task itself, and critics in PPO-on-LLMs turn out to be quite inaccurate (VinePPO, Kazemnejad et al., 2024). For single-turn tasks with verifiable outcome rewards, a 64-sample Monte Carlo mean is an *unbiased* per-question value estimate with no approximation error. It also saves memory and compute, since there's no second policy-sized model to train. So in the setting GRPO was built for (math with checkable answers), it's arguably *better* than the critic it replaces. The assumptions matter once you leave that setting.

### When the assumptions break

| Situation | Assumption broken | What goes wrong | Typical fix |
|---|---|---|---|
| Questions too easy or too hard (pass rate ≈ 1 or ≈ 0) | 1: variance | Zero advantage, wasted rollouts, effective batch shrinks as training goes on | DAPO's dynamic sampling (drop all-same groups, refill the batch); curriculum by pass rate |
| Frontier problems with very rare success (p ≈ 1/1000) | 1, 5 | 64 samples are almost always all wrong, so no learning exactly where you want it; when one success does appear, its +7.9-ish weight amplifies a lucky guess or false positive as much as a real insight | Larger G on hard questions, hints / partial solutions (the same problem STaR's rationalization addresses in Q1), better reward checking |
| Entropy collapse or low temperature | 2: independent, diverse samples | The 64 answers are near-copies, so the group covers a tiny region and the "variance" is token-level noise; exploration dies | DAPO's clip-higher, entropy bonuses, temperature control |
| Learned or noisy reward model | 4: real differences | z-scoring blows small, meaningless score gaps up to ±1; the policy learns to exploit the RM's quirks | Verifiable rewards where possible, drop std normalization (Dr. GRPO), RM ensembles |
| Heavy std normalization across mixed difficulty | 5: equal weighting | Near-solved and near-impossible questions get outsized per-sample weights; biases what the policy optimizes for | Dr. GRPO (Liu et al., 2025) removes the std term and GRPO's per-response length normalization, which also stops long wrong answers from being penalized less per token |
| Long chains or multi-turn agent tasks | 3: one scalar is enough | One reward for a 50-step trajectory gives terrible credit assignment; you can't cheaply resample 64 times from a *mid-trajectory* state, which is exactly what a critic's per-state values provide | Process rewards, per-turn advantages, or a learned critic again |
| Stochastic environments (tools, simulators, other agents) | 4: reward variance means quality | Variance comes from environment randomness, not answer quality, so the group rewards luck | More samples, fixed environment seeds, critic or variance-reduction baselines |
| Partial-credit rewards where all 64 are bad | 1, 4 | "Better than the group" ≠ good: GRPO reinforces the least-bad failure, e.g. the answer that only got the format reward | Gate rewards on correctness; keep absolute reward information |

### The underlying point

A critic and a group baseline both answer "how good is this answer *compared with what the policy usually does*?" The critic answers by generalizing a learned function across questions and states. GRPO answers by brute-force sampling at the question level. GRPO's assumption is that **the policy's own 64 samples already contain informative contrast**: a real mix of better and worse answers, scored by a reward that tells them apart, on tasks short enough that one scalar per answer is fair. That holds best for medium-difficulty, single-turn, verifiable problems. It fails at the edges of the policy's ability (all-same groups), when exploration collapses (no diversity), with noisy rewards (spurious contrast), and on long-horizon tasks (no per-step credit).

### Summary

| Aspect | PPO learned critic | GRPO group baseline |
|---|---|---|
| Value estimate | Learned V(state), per token | Mean of 64 rewards, per question |
| Bias | Function-approximation error, often large for LLM reasoning | Unbiased for the question (on-policy) |
| Cost | Extra policy-sized model to train and store | 64 rollouts per question |
| Credit assignment | Per step, in principle | Same advantage for every token in an answer |
| Needs within-group variance? | No (but no reward differences still means no signal) | Yes; all-same groups give zero gradient |
| Sensitivity to reward noise | Moderate | High with std normalization |
| Best fit | Long-horizon, multi-turn, dense or stochastic rewards | Single-turn, verifiable, medium-difficulty problems |

**In short:** GRPO assumes the 64 samples are a diverse, on-policy, mixed-outcome sample scored by a trustworthy reward, so their mean is a good value estimate and their spread is real signal. When questions are too easy or too hard, the policy has collapsed, the reward is noisy, or the task is long and multi-step, the group stops being informative. That's what fixes like DAPO and Dr. GRPO address, and why long agentic RL often brings back a critic or step-level rewards.

## Question 3

DAPO's Clip-Higher trick lets high-probability tokens increase further than standard PPO clipping would allow, while still capping how far low-probability tokens can move, specifically because a low-probability token that keeps getting clipped down never gets the room to become a viable option again. Why does an asymmetric clip fix this better than simply lowering the learning rate would?

## Answer 3

### First, a correction to the premise

Clip-Higher works the other way around. DAPO (§3.1) raises the **upper** clip bound (ε_high = 0.28, up from 0.2) so that **low-probability** tokens can rise further. It leaves the **lower** bound at ε_low = 0.2 so that low-probability tokens aren't pushed to zero. High-probability tokens didn't need help: under standard clipping they were already effectively unconstrained.

The reason is that PPO clips the **ratio** π_new / π_old to [1 − ε, 1 + ε]. That's a *multiplicative* bound, so the largest absolute increase allowed is π_old · ε, which is proportional to how likely the token already is:

| Token | π_old | Max π_new with ε = 0.2 | Max π_new with ε_high = 0.28 |
|---|---|---|---|
| "Exploitation" token | 0.9 | 1.08, so no real cap (can reach 0.999) | same |
| "Exploration" token | 0.01 | 0.012 | 0.0128 |

DAPO also measures this directly: the tokens that actually get up-clipped have mean probability below 0.2 (Fig. 3a). So the upper clip mainly binds on rare tokens, and that's where the fix goes.

The intuition about a token "never becoming viable again" is right, but it belongs to the **lower** bound. It's why DAPO *didn't* widen ε_low, as the last section explains.

### Why symmetric clipping starves rare tokens

1. **The trust region is shaped wrong.** PPO's clip approximates a KL trust region, but it treats moves very differently from what they cost in KL. Under ε = 0.2:
   - Moving the common token 0.9 → 0.999 is **allowed** and costs about **0.37 nats** of KL(π_old ‖ π_new).
   - Moving the rare token 0.01 → 0.012 is the **most allowed** and costs about **0.0002 nats**.

   The clip permits a move roughly 2,000× larger in KL for the token that's already winning, and blocks cheap, low-risk moves for the token that might be the better option.
2. **Clipping throws away the gradient, it doesn't just slow it.** Once a token's ratio passes 1 + ε with a positive advantage, the min() in the PPO objective picks the clipped term, which has zero gradient. For the rest of the minibatch passes on that rollout batch, the rare good token stops learning while common tokens keep getting updated. The signal from an expensive rollout is discarded for exactly the tokens that needed it.
3. **This makes the rich get richer.** Every update, common tokens can grow without limit, while rare ones can grow at most 20%. Even with consistently positive advantage, getting a token from 0.01 to 0.1 takes at least 13 capped updates at 1.2× per step (at least 10 at 1.28×), and the dominant token can grow faster in the meantime. Entropy falls and the 64 GRPO samples become near-identical. That's the entropy collapse in DAPO's Fig. 2b, and it starves the group baseline from Q2 of the variation it depends on.

### Why lowering the learning rate doesn't fix it

- **The learning rate is one number for every token, so the bias stays.** Scaling every step by the same factor keeps the *relative* speeds the same: common tokens still outgrow rare ones by the same proportions. You get roughly the same path toward collapse, just more slowly. Clip-Higher changes the **shape** of the constraint, so rare tokens get more room relative to common ones. A learning rate can only change its **size**.
- **A lower learning rate does mean the clip binds less often, but it gets there by moving everyone less.** That's real, and it's the strongest case for the alternative. But in the small-learning-rate limit you just get unclipped policy gradient, and that also reduces entropy on its own. The change in entropy is roughly proportional to −Cov(log π, A): raising tokens that are both likely and rewarded sharpens the distribution (Cui et al., 2025, "The Entropy Mechanism of RL for Reasoning LMs"). So the collapse comes from the update direction, not only the clip, and a smaller step size doesn't change direction.
- **It slows down the learning you do want.** In LLM RL, generating rollouts is the dominant cost. A lower learning rate gets less learning out of each expensive batch, for *every* token, including the confident correct ones. That costs sample efficiency, which DAPO is trying to *improve* (its dynamic sampling exists for the same reason).
- **It treats increases and decreases the same way.** A lower learning rate slows rare tokens going up and going down equally. It can't express the policy you actually want: "let promising rare options grow, but don't let rare options die quickly." Only an asymmetric bound can.

Raising ε_high is, in effect, a **targeted learning-rate increase for one group of tokens**: low-probability tokens with positive advantage. That's because they're the only ones the upper bound actually constrains. Everything else trains exactly as before.

### Why asymmetric, not just a wider symmetric clip

Widening both bounds would also loosen ε_low. With negative advantage, tokens could then drop faster per update (e.g., 0.72× instead of 0.8×). Repeated over many updates, rare tokens get pushed to essentially zero, and on-policy RL can only learn about tokens it *samples*. A token at zero probability is never sampled, never gets a positive advantage, and never comes back. DAPO says exactly this: increasing ε_low "will suppress the probability of these tokens to 0, resulting in the collapse of the sampling space." So:

- **Upper bound raised:** rare tokens that turn out to be good can grow quickly enough to become real options.
- **Lower bound kept tight:** rare tokens that look bad *right now* fade slowly, so they stay available for later exploration.

That asymmetry is a built-in bias toward keeping options open. A learning rate is a single scalar and can't express it.

### Caveats

- Clip-Higher is a heuristic, and ε_high = 0.28 was tuned. Set it too high and you lose the stability PPO's clip exists to provide.
- It amplifies *any* rare token with positive advantage, including one that was just in a lucky sample. With noisy rewards, that means more noise reaches rarely used options.
- It slows entropy collapse rather than eliminating it. Later work targets the covariance term directly instead of the clip, e.g. Clip-Cov / KL-Cov (Cui et al., 2025).

### Summary

| Knob | What it changes | Rare, good token | Common token | Cost |
|---|---|---|---|---|
| Symmetric clip, ε = 0.2 | Ratio bound proportional to π_old | Capped at +20% per update, then gradient zeroed | Effectively uncapped | Entropy collapse |
| Lower learning rate | Every step scaled down equally | Rises more slowly | Rises more slowly | Same direction toward collapse; worse sample efficiency |
| Wider symmetric clip | Both bounds loosened | Rises faster, **but drops faster too** | Uncapped | Rare tokens pushed to 0 and lost for good |
| **Clip-Higher** (ε_low = 0.2, ε_high = 0.28) | Only the ceiling raised | Rises faster; still drops slowly | Unchanged | Some stability risk; tuned value |

**In short:** The problem with symmetric clipping is the *shape* of the trust region. Because it's multiplicative, it gives the most room to tokens that are already likely and the least to the rare ones that exploration depends on. A learning rate is one global scalar, so it can only slow everything down equally, and the policy still heads toward collapse along the same path. Clip-Higher changes the shape: it loosens the bound exactly where it was binding (rare tokens with positive advantage) and keeps the floor tight so rare options aren't driven to zero. It's a targeted, one-directional adjustment that no single learning rate can reproduce.
