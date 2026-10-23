# The Full Lifecycle of Skills: Where Agents Are Heading, and How to Balance the Risks

## Question

As we move beyond fixed training weights, what does the "Full Lifecycle of Skills" suggest about where autonomous agent development is heading, and how should we balance this capability with the inherent risks of that direction?

## Answer

### What the "Full Lifecycle of Skills" means

A traditional LLM's abilities are **fixed at training time**: everything it can do lives in its weights. Getting a new ability means retraining or fine-tuning, which is slow, expensive and done by the model's developers.

The skills lifecycle treats abilities as **separate pieces outside the weights**: instructions, code, tool wrappers and worked examples that the agent can manage over time:

1. **Discover / Acquire**: notice a gap ("I keep failing at X") or a repeated pattern worth capturing.
2. **Create**: write the skill as a prompt, script, tool or workflow, often by reflecting on a successful run.
3. **Validate**: test it against verifiers, unit tests or evaluation tasks before trusting it.
4. **Store / Index**: save it to a skill library with a description so it can be found later.
5. **Retrieve / Compose**: load only the relevant skills when needed, and combine simple skills into more complex ones.
6. **Refine**: update the skill based on feedback from real use (failures, edge cases, environment changes).
7. **Share / Retire**: pass good skills to other agents or users, and deprecate ones that are outdated or harmful.

Examples include Voyager's growing code-skill library in Minecraft, packaged "Agent Skills" that load on demand, and tool-making agents that write their own helper functions.

### Where this suggests agent development is heading

1. **From static models to systems that keep learning.** Ability increasingly comes from the **agent system around the model** (memory, tools, skill libraries, feedback loops), not from the weights alone. Agents get better *after deployment*, during use, without gradient updates.

2. **From hand-written tools to self-written tools.** Today humans write most tools and prompts. The direction is toward agents writing, testing and maintaining their own: the agent becomes its own toolsmith.

3. **Ability that compounds.** Skills build on other skills. An agent that can reliably combine what it has learned can take on longer and more open-ended tasks, which ties directly to self-improvement (verifiers, feedback loops, search) and test-time scaling.

4. **Specialisation without retraining.** One general base model plus different skill libraries can serve as a legal agent, a DevOps agent or a research agent. Specialisation becomes cheap, fast and reversible.

5. **Shared knowledge across agents.** Skills can be shared like software packages, so what one agent learns can spread to many agents. That brings both a network effect and a supply-chain problem.

6. **Inspectable, editable behaviour.** Unlike knowledge spread through billions of weights, a skill is a **readable file**. That makes behaviour easier to inspect, version, diff and roll back, which helps safety if we take advantage of it.

### The inherent risks

| Risk | Why the skills lifecycle makes it worse |
|---|---|
| **Drift and loss of control** | The agent's behaviour changes after deployment, so what was evaluated is no longer exactly what is running. |
| **Errors that compound** | A subtly wrong skill that passes weak validation gets reused and built on, so the error spreads. |
| **Reward hacking / verifier gaming** | Self-created skills tuned against an imperfect verifier can learn to *pass the test* rather than *do the task*. |
| **Security and supply chain** | Shared or downloaded skills are effectively code plus instructions, so they can carry prompt injections, data theft or malicious tool calls. |
| **Gaining more power** | An agent that writes its own tools could give itself more access (credentials, network, permissions) than intended. |
| **Accountability gaps** | When a self-written skill causes harm, it's unclear who wrote it, who approved it, and which version ran. |
| **Library bloat and conflicts** | Large libraries collect redundant, stale or contradictory skills, which lowers reliability and makes retrieval less precise. |

### How to balance capability against risk

The guiding idea: **let agents learn freely in safe, contained settings, and require verification and human approval before anything new reaches real systems.**

1. **Treat skills like production code.**
   - Version control, code review and changelogs for every skill.
   - Tests and evaluation suites that must pass before a skill is promoted.
   - Clear provenance: who or what created it, from which run, approved by whom.

2. **Separate the learning environment from the real one.**
   - Skills are created and tested in sandboxes with no production access.
   - Promotion follows **sandbox → staging → production**, with checks at each step.

3. **Least privilege, enforced outside the agent.**
   - Skills declare the permissions they need, and the runtime enforces them, not the agent.
   - An agent can't give its own skills new credentials, network access or permissions.

4. **Strong verification, not just self-assessment.**
   - Use independent verifiers (tests, separate checker models, real-world outcomes) rather than the agent grading itself.
   - Adversarial and red-team evaluations to catch skills that game the metric.

5. **Human oversight scaled to risk.**
   - Low-risk skills (formatting, summarising) can be promoted automatically.
   - High-impact skills (spending money, deleting data, sending messages externally) need human approval.
   - Human approval for irreversible actions stays in place no matter how capable the agent becomes.

6. **Watch skills in use and be able to undo them.**
   - Log which skills ran, with what inputs and outputs.
   - Detect drift and regressions, roll back quickly, and maintain a kill switch.
   - Regularly audit the library to remove stale, duplicate or unused skills.

7. **Treat skill sharing like a software supply chain.**
   - Only accept skills from signed, trusted sources, and scan them for injected instructions.
   - Never run third-party skills with more trust than their source has earned.

8. **Re-evaluate continuously.**
   - Because the system changes after deployment, a one-time pre-launch evaluation isn't enough. Evaluate the **current skill set** on an ongoing basis against capability and safety benchmarks.

## Summary

The Full Lifecycle of Skills points to a shift from **models with fixed abilities** to **agent systems that keep learning**: they find, build, test, reuse and refine their own abilities after deployment. Ability will compound and specialisation will get cheap, while the base weights become just one part of a larger, evolving system.

That same openness creates the main risks: behaviour drift, errors that compound, verifier gaming, supply-chain attacks and agents expanding their own access. The balance comes from **treating skills as governed software**: learn in sandboxes, require independent verification and staged promotion, enforce least privilege from outside the agent, scale human oversight to the stakes, and keep everything versioned, monitored and reversible. Skills are readable files rather than opaque weights, so if we use that visibility well, this direction can make agents both **more capable and easier to control**.
