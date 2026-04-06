import os
import random
import csv

# Ensure clinical_notes directory exists
os.makedirs("clinical_notes", exist_ok=True)

# Sample flare and no-flare sentences
flare_sentences = [
    "Patient reports severe joint pain and morning stiffness.",
    "Inflammatory markers elevated, symptoms worsening.",
    "Fatigue, rash, and fever noted. Suspect disease flare.",
    "Worsening symptoms over past week, pain in multiple joints.",
    "Lupus symptoms reappeared including photosensitivity and fatigue.",
]

no_flare_sentences = [
    "Patient doing well, no complaints at this time.",
    "Labs stable, no evidence of active disease.",
    "Continues current medications without side effects.",
    "Exam unremarkable, no new symptoms.",
    "Remains in remission, plan for routine follow-up.",
]

# Generate 50 records
rows = []
for i in range(1, 51):
    label = random.choice(["flare", "no flare"])
    if label == "flare":
        text = random.choice(flare_sentences)
    else:
        text = random.choice(no_flare_sentences)

    filename = f"note{i}.txt"
    filepath = os.path.join("clinical_notes", filename)

    # Write clinical note text file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    # Add to CSV rows
    rows.append([filename, label])

# Write CSV
with open("labels.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print("Generated 50 clinical notes and labels.csv")
