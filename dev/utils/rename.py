import os
import sys
import shutil

for root, dirs, files in os.walk('modiff/results'):
    for dir in dirs:
        if "45" in dir:
            shutil.rmtree(os.path.join(root, dir))
    # for dir in dirs:
    #     if dir.endswith("controlnet"):
    #         shutil.rmtree(os.path.join(root, dir))
