import sys

def patch():
    with open("scripts/evaluate_context_challenge.py", "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "print(f\"Skipping {out_dir} - no checkpoints found.\")" in line:
            lines[i] = "            raise FileNotFoundError(f\"No checkpoint found in {out_dir}\")\n"
        elif "return" in line and "no checkpoints found." in lines[i-1]:
            lines[i] = ""
            
    with open("scripts/evaluate_context_challenge.py", "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    patch()
