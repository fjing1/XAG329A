# Weaver: Ensembling Weak Verifiers and Distillation

Questions on *Shrinking the Generation-Verification Gap with Weak Verifiers* (Weaver).

## Question 1

Weaver asks whether combining several weak, imperfect verifiers, each scoring responses imperfectly but with some positive signal, can close the generation-verification gap better than looking for one strong verifier, a bet that only pays off if those verifiers tend to be wrong in different, independent ways. When would ensembling weak verifiers backfire instead of helping?

## Answer 1

Ensembling backfires when **the verifiers' errors are correlated**: they fail on the same responses in the same direction. A hard or deceptive question that fools all of them the same way is one case, but correlation can come from many sources, and it can show up on easy questions too.

1. **Shared blind spots.** Verifiers are often built on similar base models and trained on overlapping data with similar reward-model recipes, so they share the same biases. They may all prefer longer, more confident or nicely formatted answers, or all accept a common misconception. Then 5 verifiers are about as good as 1, and you've paid 5 times the compute.
2. **Correlation produces false confidence.** This is the key mechanism. Weaver's weak-supervision aggregation assumes the verifiers are roughly conditionally independent given the true label. If that's false, the model counts 5 copies of the same mistake as 5 independent pieces of evidence. It ends up more confidently wrong than any single verifier would be. That's the actual backfire: the ensemble makes things worse, not just no better.
3. **Good signal can be drowned out.** If one verifier is strong and several weak ones share a bias, naive averaging or a badly estimated weighting can outvote the good one. The ensemble then does worse than the best single verifier.
4. **Near-chance or adversarial verifiers.** Verifiers barely above random add noise. If the weights are estimated badly, for example from too little or unrepresentative data, a below-chance verifier can even get positive weight.
5. **The generator exploits shared weaknesses.** If you optimize against the ensemble (best-of-N, RL), the generator finds the outputs that fool all the verifiers at once. Correlated blind spots then get amplified rather than averaged out. This is reward hacking.

Short version: ensembling backfires when verifiers' errors are correlated (shared training data, base models, or biases, or a hard/deceptive question that fools them all the same way), because the aggregator then treats redundant mistakes as independent evidence and becomes confidently wrong, potentially worse than the best single verifier.

## Question 2

Weaver's distillation experiment shrinks an ensemble down from a 70B model to a 400M model and reports keeping over 97% of the accuracy gains while cutting compute by more than 99.9%, even though the expensive part, working out which verifiers to trust and by how much, already happened before distillation. Does this result make you more or less confident that verification quality mostly comes from scale itself, versus from having the right training signal?

## Answer 2

**Less confident that it comes from scale itself.** The result points to the training signal, with one important nuance.

### Why the training signal matters most

- A 400M model keeps over 97% of the gains of an ensemble that includes 70B-scale verifiers. So the ability to *apply* good verification doesn't need huge capacity. A small model can carry it once it's trained on the right targets.
- What the 400M model learned from was Weaver's aggregated, reliability-weighted scores. That's a much cleaner label than any single weak verifier gives. The quality sits in the labels, not in the student's parameter count.

### The nuance

- "The expensive part already happened before distillation." The signal was produced by running many large verifiers and fitting the weak-supervision model. So scale wasn't irrelevant. It was the tool for *finding* the signal, not for *using* it.
- Better framing: **scale helped discover the signal, the signal is what matters, and once you have it, it compresses well.** It's like a teacher-student setup. You pay for the teacher once and run the cheap student many times.

### Caveats

- **Distribution.** The student was likely tested on tasks similar to its distillation data. On new domains, where you'd need to work out again which verifiers to trust, you might need the large ensemble again.
- **97% of the gains, not 97% of the accuracy.** The leftover 3% could be the hardest cases, which might be exactly where scale matters.
- **Inherited ceiling.** The student can't beat its teacher's labels. Any errors the verifiers share get distilled in too.

Short version: it makes me less confident that verification quality comes from scale itself. A 400M model can match the ensemble once it's trained on Weaver's aggregated labels, so the quality is in the training signal. But scale was still needed to produce that signal, and the result may not hold outside the distillation distribution.
