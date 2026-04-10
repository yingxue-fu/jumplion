import shutil
import filecmp
import os
import tempfile

source_file = "README.md"
dest_file = "jumplion/gui/assets/README.md"

# Read source file and strip first 4 lines
with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
stripped_content = "".join(lines[4:])

# Write stripped content to a temp file for comparison/copying
with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
    tmp.write(stripped_content)
    tmp_path = tmp.name

try:
    # Check if destination file exists and compare against stripped version
    if os.path.exists(dest_file) and filecmp.cmp(tmp_path, dest_file, shallow=False):
        print("Files are identical. No copy needed.")
    else:
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(tmp_path, dest_file)
        print("Successfully copied to jumplion/gui/assets/README.md")
finally:
    os.remove(tmp_path)  # Clean up temp file