# o2sat.py                                          by:  Edward T Peltzer, MBARI (2007 Apr 26.)
#                                                  converted to python: Jordan Thomas (2017, May 1)
#
# CALCULATE OXYGEN CONCENTRATION AT SATURATION
#
# Source:  The solubility of nitrogen, oxygen and argon in water and
#         seawater - Weiss (1970) Deep Sea Research V17(4): 721-735.
#
# Molar volume of oxygen at STP obtained from NIST website on the
#          thermophysical properties of fluid systems:
#
#          http://webbook.nist.gov/chemistry/fluid/
#
#
# Input:       S = Salinity (pss-78)
#              T = Temp (deg C)
#
# Output:      Oxygen saturation at one atmosphere (umol/kg).
#
#                        O2 = o2sat(S,T).

def o2sat(S,T):
    
    import numpy as np


    # DEFINE CONSTANTS, ETC FOR SATURATION CALCULATION

    #   The constants -177.7888, etc., are used for units of ml O2/kg.
    
    T1 = (T + 273.15)/100;
        
    OSAT = -177.7888 + (255.5907/ T1) + 146.4813*np.log(T1) - (22.2040*T1)
    OSAT = OSAT + S * (-0.037362 + T1*(0.016504 - 0.0020564*T1))
    OSAT = np.exp(OSAT);


    # CONVERT FROM ML/KG TO UM/KG
    
    O2 = OSAT * 1000 / 22.392

    return O2