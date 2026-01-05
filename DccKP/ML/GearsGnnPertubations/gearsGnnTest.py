
# imports
import sys
# sys.path.append('../')
from gears import PertData



# constants
DIR_GEARS = "/home/javaprog/Data/Broad/GeneticsML/Gears202512"


pert_data = PertData('{}/Data'.format(DIR_GEARS)) # specific saved folder
pert_data.load(data_name = 'norman') # specific dataset name
pert_data.prepare_split(split = 'simulation', seed = 1) # get data split with seed
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128) # prepare data loader

