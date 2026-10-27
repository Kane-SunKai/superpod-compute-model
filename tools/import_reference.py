"""One-time provenance import. Not required to run the standalone project."""
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from superpod_compute.contracts import Hardware

reference=root.parent/'llmcompass-model-report/reference_runtime'
dest=root/'superpod_compute/data';dest.mkdir(parents=True,exist_ok=True)
(dest/'a100.json').write_text(json.dumps(asdict(Hardware()),indent=2)+'\n',encoding='utf-8')
source=reference/'systolic_array_model/look_up_table_16_16.csv'
unique={};raw_rows=0
for row in csv.reader(source.open()):
    key=tuple(row[:6]);value=int(float(row[6]));raw_rows+=1
    if key in unique and unique[key]!=value:raise ValueError('Source LUT conflict')
    unique[key]=value
with (dest/'array_cycles.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f,lineterminator='\n')
    for key,value in sorted(unique.items(),key=lambda x:tuple(map(int,x[0][:5]))+(x[0][5],)):
        writer.writerow([*key,value])
hashes={str(p.relative_to(reference)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,reference/'software_model/matmul.py',reference/'hardware_model/compute_module.py',reference/'hardware_model/io_module.py',reference/'hardware_model/memory_module.py']}
meta={'source_url':'https://github.com/PrincetonUniversity/LLMCompass/tree/ISCA_AE','source_commit':'not available in local reference snapshot; hashes recorded instead','license':'BSD-3-Clause','input_rows':raw_rows,'unique_cycle_entries':len(unique),'source_sha256':hashes,'bundled_lut_sha256':hashlib.sha256((dest/'array_cycles.csv').read_bytes()).hexdigest(),'transformation':'Deduplicated identical keys/cycles, omitted utilization column; no cycle values changed.'}
(dest/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'rows':raw_rows,'unique':len(unique),'data_directory':str(dest)}))
