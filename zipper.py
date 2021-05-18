#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import zipfile

all_files = [f for f in sorted(os.listdir('.')) if f.startswith(("label_", "rgb_label_", "img_"))]
list_len = len(all_files)

with zipfile.ZipFile('img.zip', 'w') as zipMe:        
    for file in all_files[:list_len//3]:
        zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

with zipfile.ZipFile('label.zip', 'w') as zipMe:        
    for file in all_files[list_len//3:2*list_len//3]:
        zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

with zipfile.ZipFile('rgb_label.zip', 'w') as zipMe:        
    for file in all_files[2*list_len//3:]:
        zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

for file in all_files:
    os.remove(file)