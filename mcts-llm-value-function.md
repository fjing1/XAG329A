# The LLM as a "Value" Function in Tree Search Planning (MCTS)

## Question

In planning via Tree Search (like MCTS), what is the role of the LLM acting as a "Value" function?

## Answer

The LLM-as-value-function **estimates how promising a partial plan or state is, without playing it out to the end**. It works like a chess engine's position evaluation: from a snapshot of the board, it judges how likely this path is to succeed.

### Where it fits in MCTS

An LLM often plays two separate roles:

- **Policy (generator):** suggests the possible next steps, which become the tree's branches.
- **Value (evaluator):** scores each resulting state, for example "0.8, likely to reach a correct solution" or "this path is a dead end."

In the four MCTS phases:

1. **Selection:** travel down the tree using UCT, which balances each node's average value (exploitation) against how rarely it has been visited (exploration).
2. **Expansion:** the policy LLM proposes candidate next steps.
3. **Evaluation:** the **value LLM scores the new node**. This replaces or shortens the classic *random rollout* to the end of the task.
4. **Backpropagation:** that score is passed back up to update the averages of every node on the path.

### Why it matters

- **Pruning:** the search spends its compute on promising branches and drops bad ones early. Without a value signal, the tree grows exponentially and the search becomes close to brute force.
- **Cheaper than rollouts:** in language tasks, a full rollout means generating a whole solution, which is expensive and slow. One value call gives an estimate right away.
- **Feedback without ground truth:** many tasks have no checkable reward until the very end, or none at all. The value function provides **intermediate** feedback at each step, similar to a process reward model.
- **Test-time scaling:** it is what lets extra inference compute turn into better answers. More search plus a good evaluator gives higher quality, as in Tree of Thoughts, RAP and LATS.

### How it's done in practice

- **Prompted self-evaluation:** "Rate how likely this partial solution leads to a correct answer: sure / likely / impossible," or a score from 1 to 10.
- **Voting / comparison:** the LLM compares sibling states and picks the most promising one.
- **Trained value / reward models:** a fine-tuned model that outputs a scalar score. These are usually better calibrated than prompting.
- **Hybrid:** combine the LLM's score with real signals such as unit test results, tool output or environment reward. This grounds the score, much like the Observation step in ReAct.

### Caveats

- **Miscalibration and overconfidence:** LLMs often score their own reasoning too highly, so the search can confidently chase a wrong branch.
- **Reward hacking:** the search will exploit any weakness in the value function, picking states that *look* good to the evaluator rather than ones that *are* good.
- **Cost:** a value call at every node adds up, which is a tradeoff between search breadth and evaluation quality.
- **Mitigations:** use a separate or trained evaluator rather than self-grading, anchor scores in external checks where possible, and mix in occasional real rollouts or verifiers.

## Summary

| Role | Question it answers | MCTS phase |
|---|---|---|
| Policy LLM | *What could be done next?* | Expansion |
| Value LLM | *How promising is this state?* | Evaluation, which feeds Selection through Backpropagation |

The policy LLM decides *what could be done next*, and the value LLM decides *which of those paths deserve more exploration*. Value estimates that are good enough make tree search on language tasks practical. When they're poor, the search confidently explores the wrong parts of the tree.
