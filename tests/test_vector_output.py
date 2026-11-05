import random,unittest
from superpod_compute.scheduler import Event,EventScheduler,output_pipeline,vector_output_pipeline

class VectorOutputTests(unittest.TestCase):
 def test_zero_vector_stage_preserves_original_output_pipeline(self):
  rng=random.Random(45)
  for buffers in (1,2):
   for port in (False,True):
    for _ in range(15):
     w,l,d=[rng.random()*1e-6 for _ in range(3)];n=rng.randrange(1,25)
     a=output_pipeline([(n,w,l,d)],buffers,buffers,shared_port=port)
     b=vector_output_pipeline([(n,0,w,l,d)],buffers,buffers,buffers,shared_port=port)
     self.assertAlmostEqual(a['duration_s'],b['duration_s'],places=14)
 def test_acceleration_matches_uncompressed_dag_with_handoffs(self):
  rng=random.Random(46)
  for buffers in (1,2):
   for port in (False,True):
    v,w,l,d=[rng.random()*1e-6 for _ in range(4)];pub=.08e-6;reuse=.04e-6;n=101
    events=[]
    for i in range(n):
     events.append(Event(f'V{i}',v,('vector',),(f'W{i-buffers}',) if i>=buffers else ()))
     events.append(Event(f'W{i}',w,('shared-write',),(f'V{i}',)+((f'F{i-buffers}',) if i>=buffers else ())))
     events.append(Event(f'P{i}',pub,(),(f'W{i}',)))
     events.append(Event(f'L{i}',l,('transfer' if port else 'L2',),(f'P{i}',)+((f'D{i-buffers}',) if i>=buffers else ())))
     events.append(Event(f'F{i}',reuse,(),(f'L{i}',)))
     events.append(Event(f'D{i}',d,('transfer' if port else 'HBM',),(f'L{i}',)))
    expected=EventScheduler().run(events)[-1]['end_s']
    a=vector_output_pipeline([(n,v,w,l,d)],buffers,buffers,buffers,publish_latency_s=pub,reuse_latency_s=reuse,shared_port=port)
    b=vector_output_pipeline([(n,v,w,l,d)],buffers,buffers,buffers,publish_latency_s=pub,reuse_latency_s=reuse,shared_port=port,accelerate=False)
    self.assertAlmostEqual(a['duration_s'],expected,places=13)
    self.assertAlmostEqual(a['duration_s'],b['duration_s'],places=13)

if __name__=='__main__':unittest.main()
