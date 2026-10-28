# Scaling Test-Time Compute: Sequential vs. Parallel Search and Model-Relative Difficulty

Questions on *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* (Snell et al., 2024).

## Question 1

The scaling test-time compute study finds that easy questions do best with pure sequential revision, while hard questions need a mix of sequential and parallel search, and sequential revision only works if the model can actually recognize and fix its own near-miss. Why might that pattern flip, and what would you guess happens if the model is simply too weak to revise its own mistakes?

## Answer 1

The pattern isn't a rule about easy versus hard questions. It depends on **how far the model's first attempt is from the right answer**, and **whether the model has a reliable signal telling it what's wrong**. On MATH with PaLM 2, both of those happen to line up with difficulty. When they stop lining up, the pattern flips.

### Why the pattern exists in the first place

- **Revision is local search.** It takes an attempt and nudges it nearby. The revision model was fine-tuned on pairs of a wrong answer and a correct answer that were *close by edit distance*. So it was trained specifically to fix near-misses.
- **Parallel sampling is global search.** Each new sample is a fresh draw that can land on a completely different approach.
- **On easy questions the first attempt is usually nearly right**, so a local fix is the efficient move. Parallel samples waste compute re-deriving what you already almost had.
- **On hard questions the first attempt is often the wrong approach entirely.** Revising inside a wrong approach polishes a dead end. You need parallel samples to find a promising starting point, then revision to refine it. That's why the mix wins.

### When it could flip

1. **Hard problems where the error is a near-miss.** Picture a long derivation with one arithmetic slip, or code that's correct except for an off-by-one. That's hard by pass rate but local in structure, so pure sequential revision should win.
2. **Revision with external feedback.** With a compiler, unit tests or a proof checker, revision gets real error information instead of guessing. Sequential refinement then scales to hard problems; self-debugging loops in code are the usual example. The paper's revisions were feedback-free, which is why they depended so much on the model spotting its own mistakes.
3. **Easy problems with an unreliable reviser.** The paper found about 38% of correct answers were changed to wrong ones during revision. It only worked because a verifier or majority vote picked from across the whole revision chain, not just the last answer. If revisions are noisy, parallel samples plus a vote can beat sequential even on easy questions.
4. **Low-diversity samplers.** If the model is mode-collapsed (heavily RL-tuned, or run at low temperature), parallel samples are near-duplicates and give little exploration. An explicit "try a different approach" revision may then explore better than parallel sampling does.
5. **Long revision chains.** The reviser was trained with up to 4 prior attempts in context. Hard questions that need many rounds push it off-distribution, and gains level off.
6. **A weak or exploitable verifier.** Parallel search depends on picking the right sample. On hard questions the most common answer is often wrong, and a weak process reward model (PRM, which scores each reasoning step) gets gamed by search. That makes parallel search less valuable and changes the balance.

### If the model is too weak to revise its own mistakes

My guess:

- **The best ratio moves all the way to parallel.** A reviser that can't find its errors is basically resampling *conditioned on its own wrong answer*. That's worse than an independent sample: it anchors on the mistake, so its errors are correlated and diversity drops. It can also talk itself out of correct answers.
- **Sequential becomes a worse version of parallel.** Because the paper selects across the whole chain, a weak reviser probably won't drop below its first attempt. But the chain is a set of correlated samples, which is strictly worse than the same number of independent ones. Huang et al. (2023, "LLMs cannot self-correct reasoning yet") found the same thing: self-correction without external feedback often hurts.
- **Then everything hinges on coverage and the verifier.** Parallel search only helps if the right answer is somewhere in the model's samples *and* something can pick it out. A weak model often fails the first condition on hard questions. Its verifier, trained on the same weak model's outputs, often fails the second. When both fail, extra test-time compute of any kind gives almost nothing.
- **That's the paper's hardest-tier result, applied everywhere.** On the hardest MATH questions neither strategy helped much, and spending the compute on pretraining a bigger model beat spending it at inference. A model too weak to revise is in that regime for a much wider range of questions. Test-time compute can draw out what a model can almost do, but it can't supply what the model can't do at all.

Short version: sequential revision pays off only as far as the model can *notice* its errors, either on its own or through outside feedback. Without that, the best you can do is parallel sampling plus a good verifier. If even the verifier is weak, a bigger model is the answer.

## Question 2

The compute-optimal strategy for scaling test-time compute depends on knowing how hard a question is for the specific model, and the authors approximate that using the model's own pass@1 rate on the question, binned into difficulty quantiles, so difficulty ends up defined relative to the model rather than fixed to the problem itself. What's appealing about that definition, and where might it break down if you had to estimate difficulty before ever running the model?

## Answer 2

Defining difficulty relative to the model measures exactly what the compute-allocation decision depends on. The problem is that measuring it costs the very compute you're trying to allocate, and the estimate is weakest where mistakes are most expensive.

### What's appealing about it

- **It measures the right thing.** The strategy choice depends on where the *model's* answers fall relative to the correct one, not on how hard a human finds the problem. A question humans rate hard might be routine for the model, and the reverse. Pass@1 (how often a single sample is correct) measures model-relative difficulty directly. MATH's own human-assigned levels 1–5 only roughly track it.
- **It connects to the mechanism.** High pass@1 roughly means the model's samples land near the answer, so revising one is a good bet. Low pass@1 means most samples are in the wrong place, so you need broader exploration. The definition lines up with *why* sequential versus parallel matters.
- **It needs no human labels and adapts automatically.** Quantile bins always split the data into balanced groups for any model. You can re-run the analysis on a new model without anyone re-annotating anything.
- **It's honest that difficulty isn't fixed.** The same problem genuinely is easy for one model and hard for another, and the right policy should reflect that.

### Where it breaks down if you have to estimate it first

1. **Measuring it costs compute.** The oracle version used ground-truth correctness over many samples per question (2,048, if I'm remembering the setup right). The "predicted" version swapped ground truth for verifier scores but still averaged over that many samples. As far as I recall, the authors didn't count this cost and left cheaper difficulty prediction as future work. Spending thousands of samples to decide how to spend a few dozen makes no sense in deployment.
2. **Cheap estimates are noisy where it matters most.** With 4–8 samples, a pass@1 estimate has large error bars. Bin boundaries blur, and the hardest distinction, between 0% and a few percent, is the least resolvable. That's exactly the "will extra compute help at all?" decision.
3. **The verifier is least reliable on hard questions.** Process reward models (PRMs) tend to be confidently wrong on solutions that look plausible but are wrong. A hard question can get labeled easy, which sends it down the pure-revision path, where it gets stuck in the wrong approach. The costs are lopsided: calling an easy question hard wastes a little compute, while calling a hard one easy wastes the whole budget.
4. **Quantiles don't work for one question at a time.** Bins are quantiles *of a dataset*. At deployment you see one question and need absolute thresholds. Those thresholds come from the benchmark's distribution and may not carry over to real traffic.
5. **Pass@1 is a proxy for the thing you actually care about.** Two questions with 10% pass@1 can fail very differently. One always picks the wrong approach; the other has the right approach with one fragile step. The first needs parallel exploration and the second needs revision, as discussed in Question 1. A single number merges two failure modes that call for opposite strategies.
6. **The starting point isn't the payoff.** The allocation decision really depends on *how much accuracy improves per unit of compute*, not on accuracy at one sample. Two questions with the same pass@1 can have very different pass@k curves (chance that at least one of k samples is correct). The 0% bin also lumps together "impossible for this model" with "rare but reachable," which deserve opposite decisions: don't spend versus spend heavily.
7. **It has to be redone for every model and setup.** Any fine-tune, prompt change or temperature change shifts pass@1, so difficulty labels can't be reused. It also assumes a checker exists. For open-ended tasks with no answer key, pass@1 isn't even defined.

### What a practical version might look like

Treat difficulty estimation as part of the search rather than a step before it:

- **Probe, then commit.** Spend a small budget, say 4 samples, and look at cheap signals such as how much the samples agree and how spread out the verifier scores are. Then pick a strategy, or stop early. This is basically a bandit or early-stopping problem.
- **Learn a predictor from the prompt alone.** Train a lightweight probe on the model's hidden states for the question. That costs only one forward pass over the prompt, with no generation. Because it reads the model's own activations, it keeps the model-relative property.
- **Predict the payoff, not the starting point.** Train the predictor to estimate expected improvement from extra compute rather than pass@1.

Overall, the model-relative definition is the right target, but the pass@1-quantile version only works as an after-the-fact analysis tool. In deployment you need a cheap, calibrated estimate of how much extra compute will help each question.
