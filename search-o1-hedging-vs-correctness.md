# Search-o1: Does Less Hedging Mean More Correct Reasoning?

Question for Part 2, §3.3 (Q3), on *Search-o1: Agentic Search-Enhanced Large Reasoning Models* (Li et al., 2025; arXiv 2501.05366). The PDF in `part2/` is a one-page print of the project website. Details below come from the full arXiv paper.

## Question 3

Search-o1's evaluation measures knowledge gaps indirectly by counting hedging words like "perhaps" and "alternatively" in a model's reasoning. It finds these appear 30 or more times per output on a hard benchmark, and shows Search-o1 cuts that down substantially by letting the model trigger its own searches mid-reasoning. But sounding less unsure isn't automatically the same thing as being more correct. Is a drop in hedging words good evidence that the reasoning got more correct, or could a model just become more confidently wrong instead?

## Answer 3

### First, what the paper actually shows

The paper **doesn't** report a drop in hedging for Search-o1. The hedging count appears only once, as **motivation**: Figure 1 counts "high-frequency uncertain words per output" for the *baseline* QwQ-32B-Preview on GPQA diamond, with "perhaps" alone averaging over 30 per reasoning process. The paper never re-counts those words on Search-o1's outputs. Its own case studies still contain them, e.g. "Alternatively, perhaps the initial trans-cinnamaldehyde has more or fewer carbon atoms than I thought."

The evidence for Search-o1 is **final-answer accuracy** (Pass@1). On GPQA diamond (Table 1):

| Method | Physics (86 Qs) | Chemistry (93) | Biology (19) | Overall (198) |
|---|---|---|---|---|
| QwQ-32B, direct reasoning | 75.6 | 39.8 | 68.4 | 58.1 |
| RAG-QwQ-32B | 76.7 | 38.7 | 73.7 | 58.6 |
| RAgent-QwQ-32B (agentic search) | 76.7 | 46.2 | 68.4 | 61.6 |
| **Search-o1** | 77.9 | 47.3 | 78.9 | **63.6** |

So the paper's structure is: hedging is a *symptom* that motivates the method, and accuracy is the *outcome* it's judged on. That's the right structure. The question still matters, though, because "search reduces uncertainty, and so improves reasoning" is the paper's *explanation* for why the method works, and that explanation is never tested directly.

Two caveats on the accuracy numbers themselves:
- **The benchmark is small.** The overall gain is about **11 more questions out of 198**, against a standard error of roughly ±3.4 points for each method. The paired difference is more reliable than that suggests, but it's still modest evidence.
- **The subject breakdowns are noisier.** The Biology jump (68.4 → 78.9) is **2 questions out of 19**.

### Why a drop in hedging alone would be weak evidence

Hedging words measure **how the reasoning is worded, not what the model knows**. Several mechanisms can make hedging fall without making the reasoning more correct.

1. **Retrieved text sounds authoritative, whether or not it's right.** Once a document is in context, the model tends to treat it as fact. If search returns something wrong, irrelevant, or out of date, the model can stop hedging *and* be wrong. That's exactly "confidently wrong." RAG studies such as ClashEval (Wu et al., 2024) find that models often adopt incorrect retrieved content even when their own prior knowledge was right.
2. **Reason-in-Documents strips out the source's uncertainty.** Search-o1 doesn't insert raw documents. A module condenses them into short, declarative "knowledge" steps. Summarizing tends to drop caveats, so a hedged or disputed source turns into a flat statement. The paper's own GPQA case shows this. The condensed search result says later steps "do not change the carbon atom count." The model then (correctly, as far as I can tell: the final step is a cyclopropanation that adds a CH₂) overrides it and gets 11 carbons and answer B. So the injected "fact" was wrong and was stated with full confidence. Here the model got lucky by ignoring it. A model that trusted it would have been confidently wrong, and a hedge counter would have scored that run as *better*.
3. **Hedge markers in reasoning models are also search moves.** In o1/R1-style models, "wait," "alternatively," and "perhaps" are largely learned **exploration and self-check moves**. They're where the model branches, reconsiders, and catches its own errors (the R1 "aha moment"). Fewer of them can mean *less self-verification*, not more knowledge. Compare entropy collapse in Q3 of [train-time-scaling-rl.md](train-time-scaling-rl.md): a less exploratory policy sounds more certain and can be worse.
4. **Shorter traces mean fewer hedges anyway.** If inserted knowledge shortens or reorganizes the reasoning, raw hedge counts per output drop even if the hedge *rate* doesn't change. Counts have to be normalized per token, at least.
5. **The register changes.** Long blocks of inserted reference-style text push the model toward a summarizing, declarative style. That changes word frequencies without changing epistemic state.

The underlying issue is that **hedging is only valuable when it's calibrated**. A model that hedges exactly when it's likely to be wrong is *more* trustworthy than one that never hedges. The goal is to remove *unwarranted* uncertainty, where the model could have known, while keeping hedges that signal real risk of error. A lower overall count can't tell those two apart.

### What would count as good evidence

Count hedges *conditioned on correctness*, and check that the drop happens for the right reasons:

| Test | What a real improvement looks like | What "confidently wrong" looks like |
|---|---|---|
| **Hedges × correctness**, baseline vs. Search-o1 | Hedging falls mainly on questions that went from wrong to right. Wrong answers stay hedged. | Hedging falls evenly, including on wrong answers |
| **Hedging as an error predictor** (AUROC of hedge rate for predicting a wrong answer) | Stays the same or improves | Gets worse: confidence no longer tracks correctness |
| **Hedge sites in the baseline** | Search-o1 searches at the knowledge points where the baseline hedged, and the post-search claim is *true* when fact-checked | Searches happen, but the retrieved or condensed claims are often wrong or unused |
| **Noisy-retrieval ablation** (inject wrong or irrelevant documents) | Hedging stays or rises, and accuracy falls gently | Hedging *still* drops. It tracks "having a document," not "being right." |
| **Counterfactual document swap** | The answer changes when the key fact is changed (the answer actually depends on the evidence) | The answer doesn't change (search was decoration), or follows a planted falsehood without resistance |
| **Calibration of stated confidence** | Lower expected calibration error (ECE), better accuracy when allowed to abstain | Confidence rises faster than accuracy |
| **Errors by type** | Fewer errors on knowledge facts, while reasoning errors stay about the same | More errors of the "trusted a bad source" kind |

The paper's case studies hint at both outcomes. In the HotpotQA case, search **resolves** a real conflict ("Earlier, I had Chino, California"; the retrieved text clarifies "born in Chino... but formed the band in Toluca Lake"), and the model ends with "I can confidently say...". That's hedging replaced by evidence, the good case. In the GPQA case, the right answer comes despite an inconsistent chain and a wrong retrieved claim. That's a correct label with reasoning nobody should trust. **Checking only the final answer can't tell these two cases apart**, and a hedge count can't either.

### Summary

| Claim | Status |
|---|---|
| Baseline QwQ hedges a lot on GPQA diamond (over 30 uncertain words per output) | Reported (Fig. 1), as motivation |
| Search-o1 reduces hedging | **Not reported.** The paper's case studies still hedge. |
| Search-o1 is more accurate | Reported: 58.1 → 63.6 on GPQA diamond (about 11 of 198 questions). Real but modest; subject splits are noisy. |
| Less hedging ⇒ more correct | **Not implied.** Authoritative retrieved text, condensing, less self-checking, and shorter traces can all lower hedging without improving correctness. |
| How to tell | Condition hedging on correctness, test it as an error predictor, fact-check claims at search sites, and run noisy-retrieval and document-swap ablations |

**In short:** A drop in hedging is evidence that the model *sounds* more certain, not that it's more often right. Retrieval makes this proxy especially unreliable, because inserted documents, especially condensed ones, sound authoritative whether or not they're true. The question to ask isn't "did hedging go down?" but "did hedging go down **where the model became right**, and does it still go up where the model is wrong?" Search-o1's paper rightly bases its claim on accuracy instead, but its explanation (search fills knowledge gaps, which reduces uncertainty, which improves reasoning) is never tested directly. Its own GPQA case study, which reaches the right answer from a wrong retrieved fact, shows why the explanation needs that test.
