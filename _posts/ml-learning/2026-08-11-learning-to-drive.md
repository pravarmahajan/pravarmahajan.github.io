I watched the virtual car in a synthetic town rendered by a video game engine follow a predetermined path to reach its destination. Then I watched it again, and again and again. I wrote an eval script which counted exactly 6 successes out of 10. My unsophisticated, RL based model has finally learned to drive!

<video controls width="100%">
  <source src="/videos/Carla First Drive.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

This is a small thing by the standards of the field. A near trivial thing, a simulated car with perfect sensors, 200 meter route, no traffic, no pedestrians. But between that first baseline model which was stalled in one place to the quiet Tuesday night I saw my simulated car finally touch the finish line, I had a lot of learnings that I would like to share in this post.

## Why I Started Teaching a Car to Drive

I live in the heart of The Silicon Valley in California. Not a day goes by without seeing those autonomous driverless Waymo driving themselves around. They have always fascinated me. I am an ML engineer by profession and I love to tweak things. I started this journey to learn how these self driving algorithms work under the hood - and pick up some reinforcement learning skills on the way.

## CARLA
[CARLA](https://carla.org/) is an open-source driving simulator built on Unreal Engine. It gives you a city, a car, sensors, physics, and a Python API that exposes everything. It's a real self-driving testbed that a person can run at home, in a Docker container, on a single machine.

The CARLA bundles itself with several examples. Before I start building my own agent, I tried a few examples to get a hang of running the simulator and making sure everything works correctly. The simultor setup itself took some time. At the time of writing this, the official carla isn't supported on Ubuntu 26.04. So I downloaded a docker image and ran the server with GPU support.

```
docker pull carlasim/carla:0.9.16

docker run --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.16 /bin/bash ./CarlaUE4.sh -vulkan -RenderOffScreen
```

Then I cloned the [CARLA repo](https://github.com/carla-simulator/carla) and tried `manual_contro.py` spending a few minutes driving around. It also contains `automatic_control.py` which has some hardcoded rules to allow the car drive around the city.

## My First RL Driving Agent

Simple setup:

- **CARLA server** (Docker, Unreal Engine) running a town and ticking in
  *synchronous mode* at a fixed 20 Hz.
- **`CarlaGymEnv`** — a `gymnasium.Env` wrapper. It spawns a vehicle at a random
  spawn point, picks a goal 100–200 m away, and uses CARLA's
  `GlobalRoutePlanner` to resolve an actual point-A-to-point-B path along the
  road graph, sampled into waypoints every 2 meters.
- **PPO from Stable-Baselines3** (`MlpPolicy`), training via `train.py`,
  inspected via `drive.py` (a pygame window showing the camera feed) and scored
  via `eval.py`.

The observation was four numbers:

```
[speed, distance to next waypoint, angle to next waypoint, distance to goal]
```

The action was two numbers: steering in `[-1, 1]` and throttle in `[0, 1]`.
There was no brake - the agent had no way to slow down. 

The reward was crude: a flat `+10` per waypoint passed, and terminal penalties
for crashing. PPO with the standard recipe: `n_steps=2048`, `n_epochs=10`,
`gamma=0.99`, and `ent_coef=0.0` — the Stable-Baselines3 default

**The first result: "baseline behavior only."** The car existed, moved, mostly
wandered and crashed. Then round 2 was a disaster — and not the fun kind.
Reward hovered around −9.98, episodes lasted 1–4 steps, and I burned ~2,700
episodes before I stopped it. The root cause had nothing to do with reward
design: the training process was executing a stale, buggy version of the
stall-detection code that was still loaded in memory from an earlier session. I
had been tuning a reward against an environment that wasn't the environment I
thought I was running.

Lesson one, learned the hard way: **before you blame the reward, verify the
execution.**

---

## 3. The Car Learned to Stand Still

Rounds 3 through 7 were steady progress. I widened the observation to six
dimensions, then eight. I added exact cause-of-death logging
(`crash` / `off_road` / `wrong_way` / `stall` / `timeout` / `success`), a
lane-invasion sensor, off-road detection, wrong-way detection, input
normalization, and a held-out deterministic evaluation. By round 7 the car
stayed in its lane and followed curves — something it had never done before.
But it still never reached the destination.

Round 8 exposed the first real pathology: **reward hacking.** Because the reward
included a flat per-tick term proportional to speed, the policy discovered it
could net enormous reward just by driving fast and surviving long — including on
episodes that ended in a crash. One episode logged `reward=4732.67,
reason=crash`. The agent was optimizing "stay alive and collect speed reward,"
not "finish the route." I removed the speed reward.

Round 9, the car learned to stand still.

I had stripped the flat speed term, stacked on penalties (steering smoothness,
lane offset, obstacle proximity), and cranked up the terminal punishments. The
result: a policy that collapsed onto the lowest-risk behavior available —
barely moving, or creeping forward a few meters while sawing the wheel back and
forth in place. The episode log's back half was a wall of `reason=stall`.

Two compounding causes. First, `ent_coef=0.0` meant no exploration pressure at
all: once the policy found a low-risk mode, nothing pushed it back out. Second,
my stall check was gameable — it only fired when `speed < 0.1` *and*
`throttle > 0.5`, so the policy could sit still while applying gentle throttle
and never trip it.

Rounds 10 and 11 patched both: an entropy bonus, a tighter stall check, unified
terminal penalties. The pure freeze went away. In its place: a jittery,
oscillating "drunk walk" that was slower and *worse* than round 8. Every round
since round 8 had made driving quality worse, not better.

Then the numbers explained everything. Pulling the reward distribution across
rounds:

| Round | Median reward | Best reward | Successes |
|---|---|---|---|
| 8 | +1168 | +5204 | 7 |
| 9 | −217 | −7.5 | 0 |
| 10 | −161 | −14.2 | 0 |
| 11 | −243 | **−148.5** | 0 |

Since round 9, **not one episode out of 554 earned positive total reward.** The
reward function had become a pure cost function. And under discounted RL with
an everywhere-negative reward, the optimal policy is to end the episode as
cheaply and as quickly as possible. Round 11's single best outcome across 230
episodes was a stall at −148.5. **Freezing immediately was literally the
argmax.**

This is the crucial thing to internalize about reward hacking: the agent isn't
cheating. It is doing exactly what you asked. If you don't like the behavior,
the objective — not the agent — is wrong.

---

## 4. Why Reward Shaping Wasn't Fixing the Problem

At this point I had spent four rounds in the classic whack-a-mole loop: observe
bad behavior, tweak a coefficient, rerun 150k steps, eyeball the car, repeat.
It was time to admit the loop itself was broken, for two separate reasons.

**Reason one: every reward term is an exploit surface.** Here is the catalog of
the first several failures, each one "fixed" by a new term that created the
next exploit:

| Fix | New exploit |
|---|---|
| Flat speed reward | Farmed by driving fast + surviving long (+4,732 on a crash episode) |
| Steering-smoothness penalty | Punished the policy's own exploration noise ~30× harder than the entropy bonus could counteract → the policy collapsed its action variance, i.e. stopped moving |
| Dashed-lane (Broken marking) penalty | Fired on every *legal* junction turn — dashed lines are mandatory to cross when turning |
| Obstacle-braking reward | Farmable by parking next to a static prop and holding the brake forever |
| Stall check | Gamed via the throttle band (0.1–0.5) that never tripped it |

**Reason two: the methodology couldn't tell me anything.** Every round bundled
two to four changes, ran once (`n=1`), and was judged by eye. There was no
fixed, reward-independent scoring metric, so "did this round help?" was never
actually measurable above run-to-run noise. I wasn't running experiments; I was
playing whack-a-mole with my own intuition and calling it training.

The reframe that finally stuck: **I was optimizing symptoms.** You don't fix a
broken objective by piling more terms onto it — every term enlarges the exploit
surface, and every new exploit looks like a new "bug" to patch. The fix had to
come from understanding the mechanics underneath (section 5) and from changing
the *structure* rather than the coefficients: rebalance the reward so competent
driving is net-positive again, delete the penalty that was fighting the
algorithm, raise the discount factor, and move the smoothness fix out of the
reward function entirely and into the control layer.

---

## 5. What Does PPO Actually Optimize?

I had been using PPO the way you use a library: as a black box that turns
reward into behavior. When the behavior went wrong, I blamed the reward.
Eventually I stopped guessing and sat down with the actual math. This section is
the part of the journey that retroactively explains everything that happened in
rounds 9–11.

PPO's loss, in one line:

```
loss = −clipped_surrogate(ratio, A) + vf_coef · value_loss − ent_coef · entropy
```

with `ratio = π_new(a|s) / π_old(a|s)`. The loop is: collect `n_steps` of
rollout, compute advantages via GAE ("was this action better or worse than
expected from this state?"), then take a few clipped gradient steps. The core
intuition: whatever action got sampled and turned out to have high advantage
gets pushed up in probability; low or negative advantage gets pushed down.

Three details turned out to matter enormously for this project.

**Detail one: sigma is not what you think it is.** For continuous actions, PPO
outputs a Gaussian per action dimension: a mean `μ(s)` *and* a standard
deviation `σ`. It's easy to assume `σ` is another output of the network. It
isn't. `μ(s)` is state-dependent, computed by a real forward pass. `σ` is a
separate, free-standing trainable parameter — `log_std`, one scalar per action
dimension — that doesn't depend on the state at all. It starts at `exp(0) = 1.0`
by default.

**Detail two: sigma is pulled in two directions at once.** Every update, two
gradient pressures compete on `σ`. Negative advantages push it down: if large
sampled deviations keep landing on bad outcomes, the policy reduces the
probability of large deviations — which for a Gaussian means shrinking `σ`.
The entropy bonus pushes it up: Gaussian entropy has a closed form,
`H = ½·log(2πe·σ²)`, monotonically increasing in `σ`. Whichever pressure is
numerically larger wins.

**Detail three: this explains the rounds 9–11 mystery.** My steering-smoothness
penalty was computed on the *sampled* action — and at `σ ≈ 1`, two consecutive
independent Gaussian samples differ by about 1.13 on average. So the penalty
was largely taxing the policy's *own exploration noise*, not genuine erratic
driving. The pressure to shrink `σ` was roughly 30× stronger than
`ent_coef=0.01`'s pressure to keep it up. The policy had exactly one way out:
collapse its action variance. Which looks like a car giving up and refusing to
move.

And then there's **gamma.** The effective planning horizon of a discounted agent
is `1/(1 − γ)` decisions. At `γ = 0.99`, that's ~100 decisions — about 5 seconds
at 20 Hz. My +500 success bonus, sitting ~1,000+ steps away at the end of a
route, had a *present value of roughly 0.02*. Mathematically invisible. The
agent wasn't ignoring the destination; the destination didn't exist inside its
objective. I wanted to fix this by jumping to `γ = 0.9999`, until I thought
through the tradeoff: an extreme gamma makes the value function's targets much
noisier, and on a ~150k-step training budget, you trade "can't see the goal"
for "value estimates are too noisy to learn anything." The moderate route —
raise `γ` to 0.995 *and* hold each action for 4 physical ticks — extends the
real-time horizon 4× without gamma's variance tax.

Lesson: I had been treating PPO's knobs as tuning parameters. They're physics.
When the behavior is weird, the math explains why — if you're willing to look.

---

## 6. My First Clean RL Experiment

Armed with the diagnosis, round 12 rebuilt the foundation in one shot, with a
reasoned change for each identified failure:

- **Reward rebalanced back to net-positive for competent driving.** Progress
  shaping up, penalties down. If per-tick reward is positive, ending an episode
  early already forfeits future reward — a sufficient disincentive against
  dying — so the huge terminal penalties (which had been actively harmful once
  per-tick reward went negative) could shrink from 150 to a uniform 30.
- **The steering-smoothness penalty was deleted entirely** and replaced with
  two structural fixes at the control layer: **action-repeat** (each decision
  held for 4 physical ticks, so the car literally cannot oscillate faster than
  5 Hz) and a **steer low-pass filter**:

  ```
  applied_steer = 0.7 · previous_applied_steer + 0.3 · raw_sampled_steer
  ```

- **`gamma` 0.99 → 0.995**, combined with action-repeat, to make route
  completion visible in the objective.
- The dashed-lane penalty reverted; the obstacle-braking reward gated so it
  can't be farmed by parking; a latent learning-rate bug fixed (resumed runs
  had been silently ignoring `--learning-rate`).

And then, critically, round 13 was a **pure continuation**: `git diff` against
HEAD was empty. No code, no hyperparameter change. The identical recipe, ~150k
more timesteps. This was the first time my methodology actually isolated a
single variable.

The results are the best thing in this entire project's history:

- **307 of 1,557 rollout episodes reached the destination — 19.7%** (round 12:
  43 of 5,296).
- Successes climbed steadily through the run: 35 / 49 / 96 / 127 per quartile,
  and 68 (34%) in the last 200 episodes. The rate was still rising when the run
  ended — the checkpoint is not saturated.
- Standalone deterministic evaluation on the final checkpoint: **6/10 routes
  completed.**

Now the discipline part — the caveats I have to repeat to myself every time I
get excited:

- `n = 1`. One run. Six of ten attempts on fresh random routes is a ±~15%
  noise estimate of the true completion rate.
- Random routes mean junction-heavy variants are over- and under-sampled
  unpredictably; I don't know the difficulty coverage.
- The training data is a long way from solved: 538 crashes, 416 off-road, 298
  wrong-way terminations during the round. The degenerate no-move mode is still
  reachable, just no longer dominant.

But here's what makes it clean: the improvement is attributable. One thing
changed — more steps on a sound recipe — and the success rate did something
monotonic and measurable. That only worked because round 12 had built the
measurement infrastructure first: exact termination reasons, a deterministic
held-out eval, input normalization. **You can't attribute an improvement you
can't measure, and you can't measure one you didn't instrument for.**

---

## 7. Why Observation Design Matters

By round 13, the agent that drives the car perceives the world as a vector of
nine numbers:

```
[speed, distance to next waypoint, heading error to next 5 waypoints (circular mean),
 distance to goal, lane offset, heading error vs. road, previous steer, previous throttle,
 distance to nearest obstacle ahead]
```

That vector was built one feature at a time, and each addition was its own
lesson in what an agent can and can't learn from what you give it.

The first lesson was about **scale**. For the first six rounds, my observation
dimensions spanned wildly different ranges — speed 0–50, distance 0–500, angles
−π..π, actions −1..1 — and I fed them raw to the network. The value loss
literally never decreased. It's a classic failure: a critic trying to fit a
value function across inputs that differ by orders of magnitude. Wrapping the
environment in `VecNormalize` (input *and* reward normalization) was the single
biggest "algorithm-unchanged" improvement I made: explained variance jumped to
~0.67 and episode lengths went from ~100–200 steps to 600–960+.

The second lesson was about **features**. The policy was weaving on straight
roads — and one cause was that a single 2-meter waypoint's placement jitter
could look like a steering-relevant angle change when it wasn't. Fix: average
heading error over the next five waypoints (a circular mean) instead of one.
The policy was oscillating and couldn't stop — partly because it had no idea
what it had just commanded. Fix: feed the previous steer and throttle into the
observation so the policy can see its own actions. These are tiny changes, and
each one changed driving quality more than any reward coefficient I ever tuned.

The third lesson was the sneakiest: **a sensor bug masquerades as a policy
bug.** For weeks, "the car doesn't turn at intersections" looked like a
learned-behavior failure. It wasn't. My wrong-way detector compared the
vehicle's heading against a lane-id sign convention recorded once at reset —
but that sign convention only holds within a single road segment, so turning
onto a different road flipped it and falsely flagged *correct* turns as
wrong-way. After one fix, junction connector lanes swung the heading error past
90° during legitimate turns. Both were bugs in the *measurement*, not the
learner — and no amount of reward tuning could have fixed them.

There's a practical taxonomy buried in these weeks that I wish I'd written down
on day one: **when an agent misbehaves, which layer is it — observation,
reward, model, or algorithm?** Most of my early "reward problems" were
observation or sensor problems wearing a reward-shaped costume.

---

## 8. From State Vectors to Pixels

*Status: not built yet. This is the next milestone, not a result.*

Here's the uncomfortable truth about a hand-engineered observation vector: it's
a featurizer written by me. It encodes what *I* decided a driver needs to know —
and it is therefore an upper bound imposed by my imagination. A lane offset, a
heading error, an obstacle distance: all are my guesses about which summaries of
the world matter.

Pixels are the alternative — hand the raw camera image to the network and let
it learn its own features. The plan, when I get there:

- Switch to SB3's `CnnPolicy` with downsampled, normalized camera input.
- Use **frame stacking** so velocity — currently a hand-fed feature — becomes
  *perceivable* from pixel motion across frames, the way it is for a human.
- Keep the reward and termination logic **byte-identical** to round 13's, so the
  observation change is the only variable in the experiment.
- Reconcile with a practical wrinkle: training currently disables CARLA
  rendering entirely (~500+ steps/sec — one of the better infra decisions in
  this project), and a camera during training reverses that cost.

Expected tradeoffs to test honestly: sample efficiency, per-step cost, credit
assignment over a much larger network, and whether a pixel policy can match — or
exceed — the vector policy's completion rate at all.

---

## 9. What Changes When the Agent Sees the Road?

*Status: companion to section 8 — the analysis to write after the camera agent
exists.*

Some of what I expect to learn, stated as hypotheses to be tested rather than
conclusions:

- **Convolutional inductive biases — locality, weight sharing, translation
  invariance — are a good structural fit for driving.** A road is a
  spatially-structured, translation-repetitive signal; a CNN is built to exploit
  exactly that, and an MLP on flattened pixels is not.
- **Pixels recover, at a cost, what the vector gave for free.** Curvature, lane
  markings, obstacles, and context that my nine numbers never carried all
  become *learned inferences* — but exact speed (a snapshot is a snapshot) and
  precise distance (depth must be inferred, not measured) are things the vector
  had directly. The agent's perception problem changes shape, not size.
- **New failure surfaces arrive with the new representation.** Lighting,
  weather, distractors, the semantic gap between "pedestrian" and "parked car"
  (geometry can't tell you), and eventually the sim-to-real cliff.
- **Measurement stays the same.** Same deterministic eval, same termination
  breakdown. No new metrics until the old ones are matched — otherwise the
  comparison isn't a comparison.

---

## 10. What I Still Don't Understand About Deep RL

It would be a lie to end this post pretending I've got it figured out. I keep a
running list of things I use fluently but don't deeply understand, and it is
the honest place to end:

- **GAE.** I use it, I know the shape of the formula, I could not derive it for
  you from scratch. That gap sits right at the heart of the algorithm that
  eventually worked.
- **SAC.** PPO got the car driving, but PPO is on-policy, discards its rollouts,
  and needed a fixed entropy coefficient babysitting it the whole way. Would
  SAC — off-policy, replay-buffered, with self-tuned entropy — have avoided
  rounds 9–11 entirely? Or introduced its own pathologies? The comparison
  experiment is on the list.
- **Why entropy helps exploration.** I know the equation and the empirical fact.
  I don't have a first-principles intuition for *why* maximizing policy entropy
  produces useful exploration rather than just noise.
- **When does PPO fundamentally fail?** I've now seen it fail in two instructive
  ways (mode collapse under a hostile reward; σ collapse under a hostile
  penalty). I don't yet know how to predict those from first principles.
- **Would round 13 reproduce?** One clean run is not a result. Three seeds, or
  an automated hyperparameter/reward search scored on a reward-*independent*
  metric, is the difference between a story and a finding.

And one personal note that belongs here. Mid-project, I wrote in my thinking
log: *"It probably isn't getting enough reward for completing. Maybe I should
increase it to 10000."* It took an honest critique from my research partner to
stop me: don't pull a number out of feeling — near the goal, a 10,000 bonus
destabilizes gradients, and the right reward scale is something you *derive*
from the discount factor, not something you pick. I was about to tune the
objective by vibes, at the exact moment the whole project had taught me not to.

That's the loop, and it's why I'm writing this down in public. The car drives
now — some of the time, on simple routes, in a simulated town. That's a
starting line, not a finish. And the reader is now, as I was a few months ago,
the narrator of round 14.

*Next in this series: the camera-based agent, the measurement of what it
changes, and whatever round 14 has to say about the questions above.*

