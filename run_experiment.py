import subprocess

for curiosity in range(1, 10):
    for confusion in range(1, 10):
        subprocess.Popen(['python3', "main.py", "--curiosity", f" {curiosity/10}", "--confusion", f" {confusion/10}", "--n_runs", " 100"])
