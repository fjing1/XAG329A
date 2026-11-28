# Large-Scale Sampling vs. Filtering and Clustering in AlphaCode

Questions for Part 2, §3.3, on *Competition-Level Code Generation with AlphaCode* (Li et al., 2022) and the *AlphaCode 2 Technical Report* (Google DeepMind, 2023).

## Question 1

AlphaCode's own framing calls large-scale sampling, up to a million samples per problem, a critical driving force behind its performance. But a filtering step then throws out roughly 99% of those samples using the problem's own example tests before clustering the rest, so most of what actually determines the final submission happens after generation. If sampling is really the critical driving force, does that mean generation is the part most worth investing more in, or is it actually the filtering and clustering built on top of it that make all that sampling usable?

## Answer 1

Neither one on its own. The two halves are **complements, not competitors**: each is worth almost nothing without the other. Generation decides **whether** a correct program exists among the candidates. Filtering and clustering decide **which** candidate gets submitted. Which one is worth investing in depends on which of those two is failing, and AlphaCode's own data shows both failing in different places.

A useful way to write it down:

  10@k solve rate ≈ pass@k × P(a correct sample is among the 10 submitted | one exists)

- **pass@k** is *coverage*: the share of problems where at least one of the k samples is correct. Generation sets this ceiling, and nothing downstream can raise it.
- **The second factor** is *selection efficiency*: how much of that ceiling the filter and clusters actually capture within 10 submissions.

### Evidence that sampling is worthless without selection

- **Without filtering, more samples don't help at all.** AlphaCode's Fig. 8 compares random picks, filtering, filtering plus clustering, and perfect selection (pass@k). In the paper's words: "Filtering and clustering clearly enable scaling, as otherwise the solve rate remains flat." Ten random picks from a million samples do about as well as ten random picks from ten samples. The log-linear scaling curve that makes "sampling is critical" true belongs to **sampling plus selection**, not sampling alone.
- **The paper's own framing joins the two.** The abstract lists the third key component as "large-scale model sampling to explore the search space, **followed by filtering** based on program behavior." The "critical driving force" line in §4.4 is immediately followed by "To make the most effective use of our samples we then apply filtering and clustering."
- **Selection pays off more as sampling grows.** In the build-up ablation (Table 8), clustering adds roughly **0** points at 10@1K (12.4% → 12.2%), **+0.7** at 10@10K, **+2.6** at 10@100K, and **+4.2** at 10@1M (24.2% → 28.4%). With few samples there's little redundancy to remove and few candidates to choose between. Selection's value is borrowed from the volume of sampling.

### Evidence that selection is worthless without generation

- **A filter can only remove; it can't create.** On about 8–10% of problems, even one million samples from the 41B model don't contain a single program that passes the example tests (Table 9: 92.3% of problems have at least one). No filter or clustering method can recover those. That's a pure coverage failure.
- **The filter's strength is one-sided.** Failing an example test proves a program is wrong (AlphaCode 2: samples that "could not have been correct"). Passing proves very little. That's why roughly 99% are rejected, and still thousands to tens of thousands of candidates remain. The filter is a cheap, high-recall *negative* check. It can't find a correct program that isn't there, and it can't tell apart the many survivors that pass the examples but fail hidden tests.

So the premise that "most of what determines the final submission happens after generation" is half right. Downstream decides *which* program is submitted. On hard problems, *whether that program is correct* was mostly decided upstream, when a correct sample was (or wasn't) generated.

### Where investment actually pays off

The two halves aren't priced the same, and "generation" really means two different levers:

| Lever | What AlphaCode shows | Marginal return |
|---|---|---|
| **More samples** (same model) | Solve rate is log-linear in k: each 10× more samples adds a roughly fixed amount (e.g. 41B: 16.9% at 10@1K → 31.8% at 10@1M, about +5 points per 10×). The paper says "the costs quickly become prohibitive," and sampling costs hundreds of petaFLOP-days. | **Worst.** Exponential cost for linear gains. |
| **Better model** (quality per sample) | "Better models have higher slopes," so they reach the same solve rate "with exponentially fewer samples." Training improvements alone took a 1B model from 6.7% to 12.4% at 10@1K (Table 8). AlphaCode 2 matches AlphaCode's 1M-sample performance with **about 100 samples, over 10,000× more sample-efficient.** | **Best on the generation side.** Shifts the whole curve. |
| **More diverse samples** | Half Python, half C++; random tags and ratings in the prompt; relatively high temperature. Random tags and ratings help "potentially by increasing diversity." | Good and cheap. A filter needs *independent* attempts. A mode-collapsed sampler (see DAPO, Q3 in [train-time-scaling-rl.md](train-time-scaling-rl.md)) gives a million near-copies. |
| **Better selection** | Clustering +2.4 points at 10@1M for 41B (31.8% → 34.2%, Table 5). That's about half of a 10× increase in samples, for the cost of a test-input generator and running candidates on CPU. AlphaCode 2 adds a learned **scoring model** to pick within clusters. Fig. 8 still shows "a large gap" between filtering plus clustering and pass@k. | **High while the pass@k − 10@k gap is large**, which it is. Capped by how much signal the verifier has. |

So the investment answer is:

1. **Don't invest in raw sample count.** It's the most expensive lever, and its returns depend on selection anyway.
2. **On the generation side, invest in model quality and diversity.** A better model raises pass@k for each sample, and it makes selection easier too: more correct samples means a bigger correct cluster. That's consistent with AlphaCode 2's filter removing only about 95% of samples instead of about 99%.
3. **On the selection side, invest wherever pass@k is far above 10@k.** That gap is accuracy you've already paid for in generation and are failing to capture. It's the cheapest accuracy available. AlphaCode 2 credits both "the policy models generating the code samples, and the scoring model used to select the best of them" as the two critical components.

Measure pass@k vs. 10@k for each difficulty level to decide. Problems with pass@k ≈ 0 need a better generator. Problems with a high pass@k and a low 10@k need a better selector.

### Why selection has its own ceiling

Clustering picks clusters **from largest to smallest**, on the assumption that "correct solutions tend to behave the same" while wrong ones scatter. That's a majority vote over program behavior, and it fails the way majority votes do:

- **A common wrong approach wins.** A typical misreading of the problem, or a brute-force solution that passes small inputs but times out on hidden ones, can form the largest cluster.
- **A rare correct solution gets missed.** *Large Language Monkeys* (Brown et al., 2024) finds that majority voting and reward models **plateau at around 100 samples** while coverage keeps rising, because "the occurrence of rare, correct solutions does not affect the most common answer." AlphaCode's 10@k curve "bending down slightly at high sample budgets," while pass@k keeps climbing, plausibly reflects the same effect.
- **The verifier comes from the domain, not the model.** Example tests exist only because competitive programming provides them. *Large Language Monkeys* shows that coverage gains turn directly into accuracy where automatic verifiers exist (unit tests, Lean proofs) and mostly stay unused where they don't. Without a cheap check, a million samples would be close to useless. This is the generation–verification gap from [weaver-weak-verifiers.md](weaver-weak-verifiers.md).

### The deeper answer

What actually drives performance isn't sampling or filtering alone. It's the **sample-and-check loop**, and that loop only exists because the problem supplies a cheap way to reject wrong programs. Sampling is the search and the filter is what makes searching worthwhile. Sampling gets called "the" driving force because it's the part that scales with compute. But the scaling curve is a property of the whole pipeline, and its slope is set by the model's quality and the selector's precision.

### Summary

| Question | Answer from the papers |
|---|---|
| Does more sampling help without selection? | No. 10@k is flat with random picks (Fig. 8). |
| Does selection help without lots of sampling? | Barely. Clustering adds ~0 at 1K samples, +4.2 points at 1M (Table 8). |
| What does generation decide? | *Whether* a correct candidate exists (pass@k). About 8–10% of problems have none even at 1M. |
| What does selection decide? | *Which* candidate is submitted. A large gap to pass@k remains. |
| Cheapest gains on the generation side? | Model quality and diversity. AlphaCode 2 is 10,000× more sample-efficient. |
| Cheapest gains on the selection side? | Better test generation and learned scoring, while pass@k ≫ 10@k. |
| Worst investment? | Raw sample count. Log-linear returns at exponential cost. |

**In short:** Sampling is the source of every correct answer, and selection is the only way to find one. Each makes the other valuable: without filtering, a million samples score like ten; without enough samples, the best filter has nothing to find. "Sampling is the critical driving force" really means "sampling plus a cheap verifier." The best returns are in making each sample better (AlphaCode 2's 10,000× gain), not in drawing more, and in closing the large gap between what the samples contain and what 10 submissions capture.

## Question 2

AlphaCode 2 reaches the same performance level with about 100 samples that the original AlphaCode needed a million samples for. It does this largely by replacing the original's fine-tuning step, and its from-scratch pretraining on GitHub, with a strong general-purpose model (Gemini Pro) plus a learned scoring model. That's a change to what the model brings to the table rather than to the sampling pipeline itself. Does this look more like progress from a better base model, or progress from a better pipeline around the same idea, and does that distinction matter?

## Answer 2

### First, what actually changed

AlphaCode 2 didn't drop fine-tuning. It replaced **pretraining**, not fine-tuning. The report: "Our starting point is the Gemini Pro model, on which we apply **two consecutive rounds of fine-tuning using GOLD**." The first round uses an updated CodeContests (about 15K problems, 30M human solutions, better tests). The second uses "a different, higher-quality dataset." So the comparison changes many things at once:

| Component | AlphaCode (2022) | AlphaCode 2 (2023) | Base model or pipeline? |
|---|---|---|---|
| Pretraining | From scratch on a 715 GB GitHub snapshot | Gemini Pro (general-purpose, web-scale) | **Base model** |
| Fine-tuning | GOLD + tempering on CodeContests | Two rounds of GOLD, on a larger CodeContests and a higher-quality set | Data and recipe |
| Policies | One model (41B, or a 41B + 9B ensemble) | A *family* of fine-tuned models, sampling budget split evenly across them | Pipeline (diversity) |
| Sampling | Python + C++, fixed temperature, random tags/ratings | C++ only, **randomized temperature for each sample**, random tags/ratings | Pipeline |
| Filtering | Example tests (removes about 99%) | Example tests (removes about 95%) | Same |
| Clustering | Behavior on generated inputs, largest first | Same, keep the 10 largest | Same |
| Picking within a cluster | Any member | Learned **Gemini-based scoring model** | Pipeline, but **built from the base model** |

The report gives **no ablation** separating these changes. So the 10,000× figure can't be cleanly attributed to any one of them. Anything below is inference from the structure of the system, not a measured split.

### The sample-efficiency claim is mostly a base-model story

There's a useful argument from how the pipeline behaves at small k:

- At **about 100 samples**, the filter (which removes about 95% on average) leaves roughly **5 survivors** per problem, which is fewer than the 10-submission budget. With that few candidates, clustering and the scoring model have almost nothing to *choose* between. You submit roughly everything that passes the example tests.
- That fits AlphaCode's own ablation (Table 8): clustering adds about 0 points at 1K samples and only pays off at 100K–1M.
- So AlphaCode 2 matching AlphaCode's 1M-sample solve rate with about 100 samples must come almost entirely from **coverage per sample**: how often one sample is correct. That's the policy, meaning Gemini Pro plus fine-tuning on it plus the diversity tricks. It isn't selection.

This is also exactly the effect AlphaCode predicted: "better models have higher slopes" and "can reach the same solve rate with exponentially fewer samples." AlphaCode 2's Fig. 3 still shows **log-linear scaling in samples**. The curve kept its shape and moved up. A better base model shifts the curve; it doesn't change the method.

The **headline** result is a different number: 43% vs. 25% at **1M samples**. In that regime there are about 50K candidates left after filtering, and the scoring model really does pick one representative from each of the 10 largest clusters. That gain plausibly comes from both, and the report doesn't separate them.

### Why the line between the two blurs

The scoring model *is* a pipeline component, but it **only exists because of the base model**. AlphaCode 1 had no learned verifier. Its "value conditioning" was a training trick, not a reranker. A reranker that can predict whether competition code is correct needs a model that already understands the code well. So a stronger base model improves **every learned part of the pipeline at once**: the policy, the test-input generator for clustering, and the scorer. Only the non-learned parts stay the same: the example-test filter and the largest-cluster-first rule. "Better base model" and "better pipeline" aren't independent: a better base model is what makes new pipeline stages possible. (Compare the generation–verification gap discussion in [weaver-weak-verifiers.md](weaver-weak-verifiers.md): the same base model gets fine-tuned into both the generator and the verifier.)

So the best description is: **the same search procedure, run with a much better proposal distribution, plus one new learned verifier that the better model made possible.** The *idea*, which is to sample massively, reject by execution, deduplicate by behavior, and pick 10, is unchanged. AlphaCode 2 still samples a million times for its headline number.

### Does the distinction matter? Yes, in five ways

1. **What to expect next.** If the gain is mostly the base model, the pipeline is a fixed scaffold, and progress will come from the next base model. The report says as much: "We suspect using Gemini Ultra ... would lead to further improvements." If the gain were mostly the pipeline, you'd expect the ideas to transfer to *other* models and domains. A team deciding where to spend should know which one it is.
2. **Real cost, not sample count.** "10,000× more sample-efficient" counts samples, not FLOPs. Gemini Pro's size isn't disclosed, and each sample from it likely costs more than one from AlphaCode's 41B model. AlphaCode 2's own discussion says the system "remains too costly to operate at scale." And Gemini's pretraining is a huge cost that isn't counted, because it's shared across every Gemini product. Economically, that sharing is the real base-model advantage: you don't train a code model from scratch. But it isn't an efficiency gain *in the method*.
3. **Whether the evaluation measures reasoning.** AlphaCode 1 used a strict temporal split: all training data on or before 2021/07/14, test problems after 2021/09/21. A general model pretrained on the web may have seen Codeforces problems, editorials, or near-duplicates. The AlphaCode 2 report evaluates on "12 recent contests" (77 problems) but doesn't describe a contamination check. If part of the gain is the base model *remembering* rather than *reasoning*, it won't carry over to truly new problems. You only need to ask this question because the gain came from the base model.
4. **Which way the field is heading.** If base models keep improving, hand-designed pipelines tend to become less important. They get absorbed into the model or replaced by the model's own reasoning. AlphaCode 2's report ends by aiming to bring its "unique capabilities to our foundation Gemini models." Later reasoning models moved further in this direction. OpenAI's *Competitive Programming with Large Reasoning Models* (2025) reports that o3, without o1-ioi's hand-designed test-time strategies, beat it. (I'm citing that from memory; it isn't in the repo.) That's the "bitter lesson" pattern. Sample-and-filter is still useful, but each generation needs less of it.
5. **What the pipeline is still for.** Even with a much better base model, the non-learned parts set the ceiling: the domain's cheap verifier (example tests) and the 10-submission budget. The report admits the system "relies heavily on being able to filter out obviously bad code samples." A better base model makes sampling cheaper. It doesn't make the method work in domains without a verifier. Keeping the two kinds of progress separate tells you which conclusions carry over to other domains: base-model gains carry over broadly, but the verifier-dependent pipeline only carries over where a verifier exists.

### Summary

| Question | Answer |
|---|---|
| Was fine-tuning replaced? | No. Pretraining was replaced. Fine-tuning stayed (two rounds of GOLD on more and better data). |
| Was the pipeline unchanged? | Mostly. Same filter and clustering. Added policy families, randomized temperatures, C++ only, and a learned scorer. |
| Source of the 10,000× sample efficiency? | Mostly the policy. At about 100 samples, about 5 candidates survive filtering, so selection has little to do. |
| Source of 43% vs. 25% at 1M samples? | Both, not separated. The scoring model matters here. |
| Is the scorer "pipeline" or "base model"? | Both. A pipeline stage that only a strong base model makes possible. |
| Why the distinction matters | Predicting next gains, true compute cost, contamination risk, the field's trend toward simpler pipelines, and which domains the method transfers to. |

**In short:** AlphaCode 2 looks mostly like **base-model progress running through an essentially unchanged search procedure**. The curve kept its log-linear shape and moved by four orders of magnitude, and at 100 samples there's barely anything left for the pipeline to select. The one real pipeline addition, the learned scorer, is itself a product of the better base model. The distinction matters because the two kinds of progress age differently. Base-model gains compound with each new foundation model and need checking for contamination. Pipeline gains depend on a domain verifier and tend to get absorbed into the model over time. Here no ablation separates them, which is itself the main caveat.
