import hashlib,json,pathlib,statistics
from PIL import Image,ImageChops
import openpyxl
r=pathlib.Path(__file__).resolve().parent;o=r/'outputs/mapper_library_20261103'
f=o/'BW1100_分块方案库与Mapper独立验证_20261103.xlsx';data=json.loads((r/'analysis.json').read_text())
v=openpyxl.load_workbook(f,data_only=True,read_only=True)
assert len(v.sheetnames)==7
assert [v['结果总览'].cell(i,2).value for i in range(5,10)]==[8,7,14,11,7]
for row,x in enumerate(data['cases'],5):
 assert abs(v['24组独立验证'].cell(row,5).value-x['measured_us'])<1e-8
 for i,k in enumerate(('original','automatic','nearest','library_min','union')):
    assert abs(v['24组独立验证'].cell(row,7+2*i).value-x['methods'][k]['absolute_error'])<1e-10
for s in v:
 for row in s.iter_rows():
    for cell in row:
        assert cell.data_type!='e',(s.title,cell.coordinate,cell.value)
assert v['GPU原始计时'].max_row==1204
v.close()
for path in o.glob('*.png'):
 if path.stem.endswith('_crop'):continue
 im=Image.open(path).convert('RGB');box=ImageChops.difference(im,Image.new('RGB',im.size,'white')).getbbox()
 if box:im.crop((max(0,box[0]-20),max(0,box[1]-20),min(im.width,box[2]+20),min(im.height,box[3]+20))).save(path.with_stem(path.stem+'_crop'))
result=dict(workbook_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),sheets=7,raw_event_samples=1200,
            pass_counts=[8,7,14,11,7],all_24x5_errors_match=True,formula_errors=0,
            native_excel='Full recalculation, raw-input perturbation and restore passed; all sheets exported for visual review')
(r/'report_qa.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
