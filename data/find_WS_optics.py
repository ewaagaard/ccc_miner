"""
Class to find optics functions at WS location: betx, bety and dx
"""
import xtrack as xt
import json
from pathlib import Path

# Calculate the absolute path to the data folder relative to the module's location
test_data_folder = Path(__file__).resolve().parent.joinpath('test_data').absolute()
seq_folder = Path(__file__).resolve().parent.joinpath('sequence_files').absolute()

class find_WS_optics:
    
        def get_beta_x_and_y_at_WS():
            """ Find betatronic functions at location of WS in Twiss table, also the horizontal dispersion"""
            
            line_SPS_Pb = xt.Line.from_json('{}/SPS_2021_Pb_ions_matched_with_RF.json'.format(seq_folder))
            line_SPS_Pb.build_tracker()
            twiss0_SPS = line_SPS_Pb.twiss().to_pandas()

            # Find wire scanner location
            betx = twiss0_SPS.betx[twiss0_SPS['name'] == 'bwsrc.41677'].values[0]
            bety = twiss0_SPS.bety[twiss0_SPS['name'] == 'bwsrc.41678'].values[0]
            dx = twiss0_SPS.dx[twiss0_SPS['name'] == 'bwsrc.41677'].values[0]

            return betx, bety, dx
        
if __name__ == "__main__":
    
    # Retrieve optics functions and store in dictionary 
    betx, bety, dx = find_WS_optics.get_beta_x_and_y_at_WS()
    optics = {'betx': betx, 'bety' : bety, 'dx': dx}
    with open('{}/SPS_WS_optics.json'.format(seq_folder), 'w') as fp:
        json.dump(optics, fp)