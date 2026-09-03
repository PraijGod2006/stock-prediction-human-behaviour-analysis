# Git Guide for Projects

This is a practical Git reference for Python projects.

Git tracks source-code changes locally. GitHub, GitLab, and Bitbucket store and share Git repositories online.

## 1. Install and configure Git

Install Git for Windows from https://git-scm.com/download/win if necessary.

```cmd
git --version
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --global init.defaultBranch main
git config --global core.autocrlf true
```

Check settings:

```cmd
git config --global --list
git config user.name
git config user.email
```

Never commit passwords, API keys, access tokens, or other secrets.

## 2. Create a repository

The Git repository belongs in the project folder, not inside the virtual environment. The environment does not need to be activated for Git commands.

For this project:

```cmd
cd /d D:\CODE\rajasthani
git init
git status
```

Recommended layout:

```text
rajasthani/
|-- .git/
|-- .gitignore
|-- README.md
|-- git_read.md
|-- pytorch.ipynb
|-- data/
|-- models/
|-- outputs/
`-- shitimon/       Virtual environment, not tracked
```

## 3. Create a `.gitignore`

Create `.gitignore` in the project root:

```gitignore
.venv/
venv/
shitimon/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
data/
models/
outputs/
*.csv
*.json
*.parquet
*.zip
*.pt
*.pth
*.onnx
.env
.env.*
!.env.example
.vscode/
.idea/
build/
dist/
*.egg-info/
.pytest_cache/
*.log
```

`.gitignore` does not affect files already tracked. Stop tracking a file while keeping it locally:

```cmd
git rm --cached path\to\file
git rm -r --cached folder-name
```

## 4. Simple recipe: finish coding and push

Use these commands after you finish a piece of work. Run them in the project folder:

```cmd
cd /d D:\CODE\rajasthani
git status
git add .
git commit -m "Describe what you changed"
git push
```

What each command means:

- `cd /d ...` moves the terminal into your project folder.
- `git status` shows changed and untracked files.
- `git add .` prepares all non-ignored changes for saving. The dot means "this project folder".
- `git commit -m "..."` saves a local checkpoint. Write a short description inside the quotes.
- `git push` uploads your committed changes to GitHub.

The first time you push a new repository, use this instead of the last command:

```cmd
git push -u origin main
```

`origin` is the name of your GitHub repository, `main` is the branch being uploaded, and `-u` remembers this connection. After that, plain `git push` is enough.

## 5. Simple recipe: create and push a branch

Use a branch when you want to work on a feature or fix without changing `main` directly:

```cmd
git switch main
git pull
git switch -c feature/my-feature
```

What these commands mean:

- `git switch main` moves you to the main branch.
- `git pull` downloads the latest changes from GitHub.
- `git switch -c feature/my-feature` creates a new branch and moves you onto it. Replace `my-feature` with a useful name, such as `data-cleaning`.

After coding on the new branch:

```cmd
git status
git add .
git commit -m "Add data cleaning"
git push -u origin feature/my-feature
```

This saves your work and uploads the branch to GitHub. The first push needs `-u`; later pushes from the same branch only need:

```cmd
git push
```

When the work is finished, open GitHub and create a pull request from your feature branch into `main`. After it is merged:

```cmd
git switch main
git pull
git branch -d feature/my-feature
```

This updates your local `main` and deletes the finished branch from your computer. It does not delete the project or your merged work.

## 6. Basic workflow

```cmd
git status
git diff
git add README.md
git add .
git diff --staged
git commit -m "Add project documentation"
git show --stat
git status
```

Use `git add .` only after checking that `.gitignore` excludes your environment and large files. A commit is a local checkpoint; `git push` uploads it.

## 7. Commit messages

Use short, specific messages such as:

```text
Add dataset loading code
Fix image preprocessing
Update training parameters
Document environment setup
```

Avoid vague messages such as `changes`, `update`, or `stuff`.

## 8. View history

```cmd
git log
git log --oneline
git log --oneline --graph --decorate --all
git show COMMIT_ID
git show --stat COMMIT_ID
git blame path\to\file.py
git log --oneline --all --grep="dataset"
git ls-files
git check-ignore -v data\dataset.csv
```

`COMMIT_ID` is the identifier shown by `git log --oneline`.

## 9. Branches

Keep `main` stable and use a branch for each feature or fix:

```cmd
git branch
```

Show local branches and branches known from GitHub:

```cmd
git branch --all
```

Create a new branch and move to it immediately. Use this before starting a feature or fix:

```cmd
git switch -c feature/data-loading
```

Move to an existing branch:

```cmd
git switch main
```

Print the branch you are currently using:

```cmd
git branch --show-current
```

Rename the branch you are currently on:

```cmd
git branch -m new-branch-name
```

Delete a local branch after its work has been merged:

```cmd
git branch -d feature/data-loading
```

Force-deleting an unmerged branch can lose its unmerged commits:

```cmd
git branch -D feature/data-loading
```

`-D` forces deletion even when the branch has unmerged commits. Those commits may become difficult to find, so use this only when you are sure you no longer need them.

## 10. Connect to GitHub

Create an empty repository online first, then connect it:

Add a connection named `origin` for your GitHub repository. Replace the example URL with your repository URL:

```cmd
git remote add origin https://github.com/USERNAME/REPOSITORY.git
```

Show the URLs currently connected to this project:

```cmd
git remote -v
```

Correct the URL for an existing remote named `origin`:

```cmd
git remote set-url origin https://github.com/USERNAME/REPOSITORY.git
```

Upload the `main` branch for the first time. `-u` remembers the connection, so later `git push` can be shorter:

```cmd
git push -u origin main
```

Upload new local commits to the remembered remote branch:

```cmd
git push
```

The `-u` remembers the default remote branch. Never put a token directly into a remote URL because it may be saved in shell history.

## 9. Clone and update repositories

Move to the folder where you want the downloaded project to be placed:

```cmd
cd /d D:\CODE
```

Download a GitHub repository, including its history, into a new folder:

```cmd
git clone https://github.com/USERNAME/REPOSITORY.git
```

Enter the downloaded project folder:

```cmd
cd REPOSITORY
```

Download information about remote commits without changing your working files:

```cmd
git fetch origin
```

Download remote changes and merge them into your current branch:

```cmd
git pull
```

`git fetch` downloads remote information without changing local files. `git pull` fetches and merges into the current branch.

A cautious update process:

Check that you have no uncommitted work before updating:

```cmd
git status
```

Download the newest information from GitHub:

```cmd
git fetch origin
```

List commits that are on GitHub's `main` but not in your current branch. If nothing prints, there are no such commits:

```cmd
git log --oneline HEAD..origin/main
```

Bring those remote commits into your current branch:

```cmd
git pull
```

## 10. Feature branch workflow

Move to `main`:

```cmd
git switch main
```

Update your local `main` from GitHub:

```cmd
git pull
```

Create a separate branch for your work and switch to it:

```cmd
git switch -c feature/my-feature
```

After editing and testing:

Check which files changed:

```cmd
git status
```

Read the exact edits that are not staged yet:

```cmd
git diff
```

Stage all non-ignored changes in the current project folder. Staging means preparing changes for the next commit:

```cmd
git add .
```

Review exactly what will go into the next commit:

```cmd
git diff --staged
```

Save the staged changes as a local checkpoint:

```cmd
git commit -m "Add my feature"
```

Upload this new feature branch to GitHub and remember its remote connection:

```cmd
git push -u origin feature/my-feature
```

Open a pull request to merge the feature branch into `main`. After it is merged:

Move back to `main` after the pull request has been merged:

```cmd
git switch main
```

Download the merged result:

```cmd
git pull
```

Delete the finished local feature branch:

```cmd
git branch -d feature/my-feature
```

Remove references to remote branches that no longer exist on GitHub:

```cmd
git fetch --prune
```

## 11. Merge and conflicts

Move to the branch that should receive the feature:

```cmd
git switch main
```

Update it before merging:

```cmd
git pull
```

Combine the feature branch into the current `main` branch:

```cmd
git merge feature/my-feature
```

Upload the merged `main` branch:

```cmd
git push
```

Cancel an in-progress merge:

Stop the merge and return to the state before the merge started:

```cmd
git merge --abort
```

Conflict markers look like this:

```text
<<<<<<< HEAD
Your current content
=======
Incoming content
>>>>>>> feature/my-feature
```

Edit the file, keep the correct content, delete every marker line, save, test, then run:

Check which files still contain conflicts:

```cmd
git status
```

Review the conflict changes:

```cmd
git diff
```

Tell Git that you fixed this file and it is ready to be committed:

```cmd
git add path\to\resolved-file.py
```

Finish the merge with a commit:

```cmd
git commit -m "Resolve merge conflict"
```

For notebooks, restart the kernel and run important cells from top to bottom after resolving conflicts.

## 12. Undo changes safely

Discard unstaged changes in one file. This cannot be easily undone:

Discard uncommitted changes in one file and restore its last committed version:

```cmd
git restore path\to\file.py
```

Discard all uncommitted changes in the project:

```cmd
git restore .
```

Unstage changes but keep them in the files:

Remove one file from the staging area while keeping its edits:

```cmd
git restore --staged path\to\file.py
```

Remove every staged file from staging while keeping all edits:

```cmd
git restore --staged .
```

Change the latest commit message before pushing:

Replace the latest commit message. Use this only before sharing the commit:

```cmd
git commit --amend -m "Better commit message"
```

Safely reverse an existing shared commit with a new commit:

Create a new commit that reverses an earlier commit. This is the safer undo method for shared branches; replace `COMMIT_ID` with the real ID:

```cmd
git revert COMMIT_ID
```

Move back one commit while keeping its changes:

Remove the latest commit from the branch but keep its file changes unstaged:

```cmd
git reset HEAD~1
```

Move back one commit and discard its changes. Use only when completely certain:

Remove the latest commit and delete its file changes. This can permanently lose work:

```cmd
git reset --hard HEAD~1
```

Do not use destructive commands on work you may need or on a shared branch.

## 13. Temporarily save unfinished work

Use `stash` when you need to change branches before your edits are ready for a commit:

Save your current uncommitted changes temporarily and label them:

```cmd
git stash push -m "Work in progress"
```

List all temporarily saved changes:

```cmd
git stash list
```

Restore the newest stash and remove it from the stash list:

```cmd
git stash pop
```

Restore the newest stash but keep a copy in the stash list:

```cmd
git stash apply
```

Delete the newest stash without restoring it:

```cmd
git stash drop
```

`pop` restores and removes the newest stash. `apply` restores it while keeping the stash.

## 14. Compare versions

Show edits in files that have not been staged:

```cmd
git diff
```

Show edits that are already staged for the next commit:

```cmd
git diff --staged
```

Compare the changes between two commits:

```cmd
git diff COMMIT_A COMMIT_B
```

Compare two branches:

```cmd
git diff main feature/my-feature
```

List only the filenames that differ between two branches:

```cmd
git diff --name-only main feature/my-feature
```

## 15. Tags and releases

Mark the current commit with a version name:

```cmd
git tag v1.0.0
```

List all local version tags:

```cmd
git tag
```

Upload one version tag to GitHub:

```cmd
git push origin v1.0.0
```

Upload every local tag to GitHub:

```cmd
git push origin --tags
```

Delete a tag locally. This does not delete the GitHub tag:

```cmd
git tag -d v1.0.0
```

## 16. Large datasets and model files

Do not commit datasets, virtual environments, checkpoints, or generated outputs to a normal Git repository. Store them on the D drive:

```text
D:\CODE\rajasthani\data
D:\CODE\rajasthani\models
D:\CODE\rajasthani\outputs
```

Track scripts that download or process data instead:

```text
scripts\download_data.py
scripts\prepare_data.py
```

Record the dataset name, source URL, download date, size, folder, and preprocessing steps in a small Markdown file. Git LFS is an option for files that must be versioned:

Enable Git Large File Storage on this computer:

```cmd
git lfs install
```

Tell Git LFS to manage PyTorch model files:

```cmd
git lfs track "*.pt"
```

Tell Git LFS to manage ZIP files:

```cmd
git lfs track "*.zip"
```

Stage the LFS rules and the model file:

```cmd
git add .gitattributes model.pt
```

Commit the LFS configuration and model:

```cmd
git commit -m "Track model with Git LFS"
```

Upload them to the remote repository:

```cmd
git push
```

Install Git LFS separately and check storage and bandwidth limits first.

## 17. Jupyter notebook advice

Before committing a notebook:

1. Restart the kernel.
2. Run important cells from top to bottom.
3. Remove huge outputs when unnecessary.
4. Save the notebook.
5. Review it with `git diff` or VS Code Source Control.

Move stable reusable functions into `.py` files and keep datasets and outputs outside the repository when possible.

## 18. Secrets and `.env` files

Never commit API keys, passwords, cloud credentials, private tokens, or private URLs containing credentials. Keep a local `.env` file ignored and provide a safe `.env.example`:

```text
DATA_PATH=D:\CODE\rajasthani\data
API_KEY=put-your-key-here
```

If a secret is committed, revoke or rotate it immediately. Removing it in a later commit does not remove it from old Git history.

## 19. Quick recovery checklist

When unsure, inspect first:

Show the current state, including changed and untracked files:

```cmd
git status
```

Show which branch you are using:

```cmd
git branch --show-current
```

Show the five newest commits in a compact format:

```cmd
git log --oneline -5
```

Show unstaged edits:

```cmd
git diff
```

Show staged edits:

```cmd
git diff --staged
```

Make a backup of important uncommitted files before destructive operations.

## 20. Reliable daily routine

Start work:

Move to this project folder:

```cmd
cd /d D:\CODE\rajasthani
```

Move to the stable `main` branch:

```cmd
git switch main
```

Get the latest `main` changes from GitHub:

```cmd
git pull
```

Create a new branch for today's work:

```cmd
git switch -c feature/todays-work
```

Before finishing:

Check your changed files:

```cmd
git status
```

Review your unstaged edits:

```cmd
git diff
```

Stage the changes you want to save:

```cmd
git add .
```

Review the staged changes one last time:

```cmd
git diff --staged
```

Create a local checkpoint with a meaningful message:

```cmd
git commit -m "Describe the work clearly"
```

Upload the feature branch and connect it to GitHub:

```cmd
git push -u origin feature/todays-work
```

Before opening a pull request:

Confirm there are no accidental uncommitted changes:

```cmd
git status
```

List commits on your current branch that are not on `main`:

```cmd
git log --oneline main..HEAD
```

Show the total changes that your branch introduces compared with `main`:

```cmd
git diff main...HEAD
```

The safest habit is: check `git status`, review `git diff`, make small commits, and use feature branches before merging into `main`.
