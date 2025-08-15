import shutil
import filecmp
import os

source_file = "README.md"
dest_file = "jumpdiaq/gui/assets/README.md"

# Check if destination file exists and compare
if os.path.exists(dest_file) and filecmp.cmp(source_file, dest_file, shallow=False):
    print("Files are identical. No copy needed.")
else:
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    shutil.copy2(source_file, dest_file)
    print("Successfully copied to jumpdiaq/gui/assets/README.md")