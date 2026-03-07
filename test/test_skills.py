import sys
sys.path.insert(0, '/data/back_home/baseline/foreblocks/agent')

import logging
logging.basicConfig(level=logging.ERROR)

import skills.02_coding.65_edit_block as edit_block
import skills.01_timeseries.80_advanced_mining as advanced_mining

print("Edit Block loaded:", dir(edit_block))
print("Advanced Mining loaded:", dir(advanced_mining))
