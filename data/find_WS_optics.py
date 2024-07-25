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
    
        def get_beta_x_and_y_at_WS(use_ions=True,
                                   ws_X = '51637', # # in 2023 we used #'SPS.BWS.41677.H/Acquisition', 'SPS.BWS.41678.V/Acquisition',
                                   ws_Y = '41677'):
            """ 
            Find betatronic functions at location of WS in Twiss table, also the horizontal dispersion
            
            Parameters:
            -----------
            use_ions : bool
                whether ion optics are used, if not Q26 protons
            ws_X, ws_y : str
                wire scanner device number used for optics
            """
            
            if use_ions:
                fname = '{}/SPS_2021_Pb_ions_matched_with_RF.json'.format(seq_folder)
            else:
                fname = '{}/SPS_2021_Protons_Q26_matched_with_RF.json'.format(seq_folder)

            line_SPS = xt.Line.from_json(fname)
            line_SPS.build_tracker()
            twiss0_SPS = line_SPS.twiss().to_pandas()

            # Find wire scanner location
            betx = twiss0_SPS.betx[twiss0_SPS['name'] == 'bwsrc.{}'.format(ws_X)].values[0]
            bety = twiss0_SPS.bety[twiss0_SPS['name'] == 'bwsrc.{}'.format(ws_Y)].values[0]
            dx = twiss0_SPS.dx[twiss0_SPS['name'] == 'bwsrc.{}'.format(ws_X)].values[0]

            return betx, bety, dx
        
if __name__ == "__main__":
    
    # Retrieve optics functions and store in dictionary 
    betx, bety, dx = find_WS_optics.get_beta_x_and_y_at_WS(use_ions=True)
    optics = {'betx': betx, 'bety' : bety, 'dx': dx}
    print('Ion optics: {}'.format(optics))
    with open('{}/SPS_WS_optics_ions.json'.format(seq_folder), 'w') as fp:
        json.dump(optics, fp)

    # Retrieve optics functions and store in dictionary 
    betx_p, bety_p, dx_p = find_WS_optics.get_beta_x_and_y_at_WS(use_ions=False)
    optics_p = {'betx': betx_p, 'bety' : bety_p, 'dx': dx_p}
    print('Proton Q26 optics: {}'.format(optics_p))
    with open('{}/SPS_WS_optics_protons_Q26.json'.format(seq_folder), 'w') as fp:
        json.dump(optics_p, fp)