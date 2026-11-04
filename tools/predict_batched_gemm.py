"""Predict an unfused batch of independent matrix products, without a GPU."""
import argparse,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from superpod_compute import Hardware,Policy
from superpod_compute.batched import predict_batched_gemm
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['batch','m','n','k']:p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--hardware',type=pathlib.Path,default=root/'superpod_compute/data/bw1100_mapper_mmac_hardware.json')
    p.add_argument('--policy',type=pathlib.Path,default=root/'superpod_compute/data/bw1100_mapper_mmac_policy.json')
    p.add_argument('--transpose-b',action='store_true');p.add_argument('--output',type=pathlib.Path)
    a=p.parse_args();v=predict_batched_gemm(a.batch,a.m,a.n,a.k,Hardware.load(a.hardware),Policy(**json.loads(a.policy.read_text())),transpose_b=a.transpose_b)
    print('duration_us='+str(v['best']['duration_us']));print('mapping='+json.dumps(v['best']['mapping']));print('output_tasks='+str(v['best']['output_tasks']));print('scope='+v['scope'])
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,indent=2))
if __name__=='__main__':main()
