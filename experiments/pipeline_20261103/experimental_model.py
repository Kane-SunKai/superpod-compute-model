"""Optional read service, not a change to default hardware or default mapper."""
from superpod_compute.operators import GemmModel
from superpod_compute.contracts import positive

class ReadServiceModel(GemmModel):
    version='experimental-directional-HBM-input-service-20261103'
    def __init__(self,read_bandwidth_Bps=None):
        if read_bandwidth_Bps is not None:positive(read_bandwidth_Bps,'read_bandwidth_Bps')
        self.read_bandwidth_Bps=read_bandwidth_Bps
    def input_hbm_bandwidth(self,hardware):
        return self.read_bandwidth_Bps or super().input_hbm_bandwidth(hardware)
