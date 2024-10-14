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
    
        def get_beta_x_and_y_at_WS(ws_device_number):
            """ Find betatronic functions at location of WS in Twiss table, also the horizontal dispersion"""
            
            line_SPS_Pb = xt.Line.from_json('{}/SPS_2021_Pb_ions_matched_with_RF.json'.format(seq_folder))
            line_SPS_Pb.build_tracker()
            twiss0_SPS = line_SPS_Pb.twiss().to_pandas()

            # Find wire scanner location
            betx = twiss0_SPS.betx[twiss0_SPS['name'] == 'bwsrc.{}'.format(ws_device_number)].values[0]
            bety = twiss0_SPS.bety[twiss0_SPS['name'] == 'bwsrc.{}'.format(ws_device_number)].values[0]
            dx = twiss0_SPS.dx[twiss0_SPS['name'] == 'bwsrc.{}'.format(ws_device_number)].values[0]

            return betx, bety, dx
        
if __name__ == "__main__":

    device_optics = {}
    
    # Retrieve optics functions and store in dictionary 
    for device_number in ['41677', '41678', '51637', '51638']:
        
        betx, bety, dx = find_WS_optics.get_beta_x_and_y_at_WS(device_number)
        optics = {'betx': betx, 'bety' : bety, 'dx': dx}
        
        device_optics['{}'.format(device_number)] = optics
        
    print('Generation of dictionary:\n{}'.format(device_optics))


    with open('{}/SPS_WS_optics.json'.format(seq_folder), 'w') as fp:
        json.dump(device_optics, fp)