from pathlib import Path

file = Path(r"\\SYNONAS-HOME\Projets\Formation Alyra\Développement IA\Projet\Rapports NLP\gpt2-xmas-finetuning_run3\checkpoint")

raw = file.read_text(encoding="latin-1")

lines = []
for line in raw.splitlines():
    if "model_checkpoint_path" in line:
        lines.append('model_checkpoint_path: "model-200"')
    elif "all_model_checkpoint_paths" in line:
        lines.append('all_model_checkpoint_paths: "model-200"')
    else:
        lines.append(line)

file.write_text("\n".join(lines), encoding="utf-8")
print("Checkpoint réparé ✔️")