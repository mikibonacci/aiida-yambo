# -*- coding: utf-8 -*-
"""helpers for many purposes"""
namelists = {'4.5':{'DIP':'_','X':'_','SE':'_'},
             '4.1':{'DIP':'_','X':'_and_IO_','SE':'_'},
             '5.x':{'DIP':'_','X':'_and_IO_','SE':'_'},
             '5.0':{'DIP':'_','X':'_and_IO_','SE':'_'},
             '5':{'DIP':'_','X':'_and_IO_','SE':'_'},
            }


def check_para_namelists(params, version):
    
    new_params = {}

    for key in params.keys():
        for level in namelists[version].keys():
            if level in key and '_CPU' in key:
                if level+namelists[version][level]+'CPU' == key:
                    pass
                else:
                    new_params[level+namelists[version][level]+'CPU'] = params[key]

            if level in key and 'ROLEs' in key:
                if level+namelists[version][level]+'ROLEs' == key:
                    pass
                else:
                    new_params[level+namelists[version][level]+'ROLEs'] = params[key]

    if new_params == {}:
        return None
    else:
        return new_params

def check_variables(params):
    
    new_params = {}

    var = {'BndsRnXp':'BndsRnXs','NGsBlkXp':'NGsBlkXs','NGsBlkXs':'BSENGBlk'}

    if 'em1s' in params['arguments']:
        for k,v in var:
            if k in params['variables'].keys() and v not in params['variables'].keys():
                new_params[v] = params['variables'].pop(k)

    if new_params == {}:
        return None
    else:
        return new_params