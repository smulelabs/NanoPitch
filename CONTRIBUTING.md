# Submitting to the NanoPitch Leaderboard

Follow these steps to submit your trained model and appear on the leaderboard.

## 1. Train your model

```bash
cd training
python train.py --data-dir ../data --output-dir ./runs/my_run \
  --epochs 50 --batch-size 32
```

Your best checkpoint will be saved at `./runs/my_run/checkpoints/best.pth`.

## 2. Create your submission directory

Create a folder under `submissions/` named after yourself (no spaces):

```
submissions/
└── your_name/
    ├── weights.pth       ← your best.pth (rename it)
    └── submission.yaml   ← required metadata
```

### submission.yaml format

```yaml
name: "Your Full Name"
note: "Brief description of what you changed — e.g. increased gru_size to 128 and tuned LR schedule."
```

Both fields are required. `note` appears on the leaderboard and in the PR comment, so make it useful.

## 3. Open a pull request

### Fork the repo (one-time setup)

If you have not already, **fork this repo** using the **"Fork"** button on GitHub. Your fork is where you should be doing all of your work and experiments — treat it as your personal copy of the project.

### Create a submissions branch

On your fork, create a branch called `submissions` (or any descriptive name). This is the branch you will use to add your submission files and open PRs against the leaderboard.

> **Important:** When opening a PR, make sure the **base repository is set to `charisrenee/NanoPitch-ClassLeaderboard`**, not the original upstream. GitHub may default to the upstream, so double-check the dropdown before submitting.

Only include the **two submission files** in your PR — `weights.pth` and `submission.yaml` inside your `submissions/your_name/` directory. Do not add training scripts, notebooks, or other experiment files to this branch.

---

### Opening a PR from VS Code

1. Make sure the [GitHub Pull Requests extension](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github) is installed.
2. Stage and commit your two submission files, then push your `submissions` branch to your fork.
3. VS Code will show a **"Create Pull Request"** prompt in the Source Control panel — click it.
4. In the PR creation view, set the **base repository** to `charisrenee/NanoPitch-ClassLeaderboard` and the base branch to `main`.
5. Title your PR and click **Create**.

### Opening a PR from GitHub Desktop

1. Commit your two submission files on your `submissions` branch in GitHub Desktop.
2. Click **Push origin** to push the branch to your fork on GitHub.
3. GitHub Desktop will show a **"Create Pull Request"** button — click it. This opens GitHub in your browser.
4. On GitHub, confirm the **base repository** is `charisrenee/NanoPitch-ClassLeaderboard` (not the upstream). Change it via the dropdown if needed.
5. Submit the PR.

### Opening a PR from the command line

```bash
# From your fork's local clone, on your submissions branch:
git add submissions/your_name/weights.pth submissions/your_name/submission.yaml
git commit -m "Add submission: your_name"
git push origin submissions

# Then open GitHub in your browser and click "Compare & pull request",
# or use the GitHub CLI:
gh pr create \
  --repo charisrenee/NanoPitch-ClassLeaderboard \
  --base main \
  --head your-github-username:submissions \
  --title "Submission: your_name"
```

---

The bot will automatically:
- Evaluate your checkpoint against the test set.
- Post your scores as a PR comment within a few minutes.

## 4. Wait for review

Once the PR is merged your scores are committed to `results/` and `LEADERBOARD.md` is rebuilt.

---

## Tips

- Only the **realtime Viterbi** score (no lookahead) counts for ranking — it matches what the browser actually does.
- Larger models aren't always better. A GRU size of 256+ can degrade due to insufficient training data.
- The biggest win is usually in the `augment_mel_batch()` stub in `train.py` — implement proper noise mixing!
- You can submit multiple times; each PR replaces your previous entry on the leaderboard.
