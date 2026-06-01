import json
import matplotlib.pyplot as plt

# Load data
with open("metrics_1layer.json", "r") as f:
    metrics1 = json.load(f)

with open("metrics_2layer.json", "r") as f:
    metrics2 = json.load(f)

# Extract values
steps1 = [x["step"] for x in metrics1]
eval1 = [x["eval_loss"] for x in metrics1]
icl1 = [x["icl_score"] for x in metrics1]

steps2 = [x["step"] for x in metrics2]
eval2 = [x["eval_loss"] for x in metrics2]
icl2 = [x["icl_score"] for x in metrics2]

# -------------------
# Eval Loss Plot
# -------------------

plt.figure(figsize=(8,5))
plt.plot(steps1, eval1, label="1 Layer")
plt.plot(steps2, eval2, label="2 Layer")

plt.xlabel("Training Step")
plt.ylabel("Eval Loss")
plt.title("Eval Loss vs Training Step")
plt.legend()

plt.savefig("eval_loss.png")
plt.close()

# -------------------
# ICL Plot
# -------------------

plt.figure(figsize=(8,5))
plt.plot(steps1, icl1, label="1 Layer")
plt.plot(steps2, icl2, label="2 Layer")

plt.xlabel("Training Step")
plt.ylabel("ICL Score")
plt.title("ICL Score vs Training Step")
plt.legend()

plt.savefig("icl_score.png")
plt.close()

print("Saved plots.")