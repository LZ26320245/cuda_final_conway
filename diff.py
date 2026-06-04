import os
import difflib

dir1 = r"output_cpu"
dir2 = r"output_gpu_v1"

for i in range(101):
    filename = f"iter_{i:03d}.txt"

    file1 = os.path.join(dir1, filename)
    file2 = os.path.join(dir2, filename)

    if not (os.path.exists(file1) and os.path.exists(file2)):
        print(f"{filename}: 檔案不存在")
        continue

    with open(file1, "r", encoding="utf-8") as f1:
        lines1 = f1.readlines()

    with open(file2, "r", encoding="utf-8") as f2:
        lines2 = f2.readlines()

    if lines1 != lines2:
        print(f"\n===== {filename} 不同 =====")

        diff = difflib.unified_diff(
            lines1,
            lines2,
            fromfile=file1,
            tofile=file2,
            lineterm=""
        )

        for line in diff:
            print(line)