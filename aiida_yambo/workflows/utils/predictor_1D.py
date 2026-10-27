import copy
import logging

import numpy as np
from scipy.optimize import curve_fit

from aiida.orm import load_node
from aiida_yambo.utils.common_helpers import find_pw_parent

LOGGER = logging.getLogger(__name__)


def create_grid_1D(edges=None, delta=None, alpha=1/3, add=None, var=None, shift=0):
    """Build a 1D convergence grid for band or k-point mesh variables.

    Creates a reference grid with strategic points (corners and midpoints) for
    convergence analysis. Used to generate fitting data without running all
    possible parameter combinations.

    Args:
        edges (list): [start, stop] endpoints for the variable.
        delta (list|int): spacing for the grid.
        alpha (float): interpolation factor for midpoints (default 1/3).
        add (list): optional extra points to append.
        var (list): variable names (e.g., ['kpoint_mesh']).
        shift (int): index offset for a shifted grid.

    Returns:
        dict: variable name -> list of grid points.
    """
    edges = edges or []
    delta = delta or []
    add = add or []
    var = var or ['BndsRnXp']

    if var[0] == 'kpoint_mesh':
        b_min = edges[0]
        b_max = edges[1]

        for i in range(3):
            b_min[i] += shift*delta[i]
            b_max[i] += shift*delta[i]
                

        A = b_min
        B = b_max
        E = []
        F = []
        for i in range(3):
            E.append(alpha*(b_max[i]-b_min[i])+b_min[i])
            F.append((1-alpha)*(b_max[i]-b_min[i])+b_min[i])

            if delta[i] == 0: 
                space_b = np.array([A[-1]])
            else:
                space_b = np.arange(b_min[i], b_max[i]+1,delta[i])

            E[i] = space_b[abs(space_b-E[i]).argmin()]
            F[i] = space_b[abs(space_b-F[i]).argmin()]

        b = []

        for i in [A,E,F,B]:
            b.append(i)

        for i in add:
            if add != []:
                b.append(i)

        return {'kpoint_mesh':b} #A,B,C,D,E,F

    else:

        try:
            b_min = edges[0]+shift*delta[0]
            b_max = edges[1]+shift*delta[0]
        except (IndexError, TypeError) as exc:
            raise ValueError(f'invalid edges or delta: {edges}, {delta}') from exc

            
        A = b_min
        B = b_max

        E = alpha*(B-A)+A
        F = (1-alpha)*(B-A)+A
        
        LOGGER.debug('%s %s', E, F)

        space_b = np.arange(A, B+1,delta[0])

        E = space_b[abs(space_b-E).argmin()]
        F = space_b[abs(space_b-F).argmin()]

        b = []

        for i in [A,E,F,B]:
            b.append(i)

        for i,j in add:
            b.append(i)

        return {var[0]:b} #A,B,C,D,E,F

class The_Predictor_1D():
    
    '''Class to analyse the convergence behaviour of a system
    using the new algorithm.'''
    
    def __init__(self, **kwargs):
            
        for k,v in kwargs['calc_dict'].items():
            setattr(self,k,copy.deepcopy(v))
        for k,v in kwargs.items():
            if k != 'calc_dict': setattr(self,k,copy.deepcopy(v))
        #print(kwargs['calc_dict'])
        
        if isinstance(self.what,list):
            self.what = self.k
        
        if not hasattr(self,'Fermi'): self.Fermi=0

        self.var_ = list(self.var)
        self.delta_ = list(self.delta)
        self.index = [0]

        # Remove GbndRnge from independent variables since it's typically enslaved to BndsRnXp
        if 'BndsRnXp' in self.var and 'GbndRnge' in self.var:
            self.var_.remove('GbndRnge')
            self.delta_.pop(self.var.index('GbndRnge'))

        if len(self.delta_) == 1:
            self.delta_ = self.delta_[0]

        # Create numpy arrays from result DataFrame columns for efficient operations
        for i in self.var_:
            setattr(self, i, np.asarray(self.result[i].values, dtype=object))

        # 'kpoint_mesh' needs its own adapter: the raw column can contain ragged
        # entries (None/NaN rows or stray scalars) that np.asarray cannot coerce,
        # so handle it before the generic extraction below.
        if 'kpoint_mesh' in self.var_:

            # Filter result rows to those with a valid 3-vector mesh so grid and
            # result stay row-aligned after sanitization.
            valid = [
                isinstance(m, (list, tuple)) and len(m) == 3
                for m in self.result['kpoint_mesh']
            ]
            if not all(valid):
                self.result = self.result[valid].reset_index(drop=True)
                if getattr(self, 'r', None) is not None:
                    self.r = np.asarray(self.r)[valid]
                for i in self.var_:
                    setattr(self, i, np.asarray(self.result[i].values, dtype=object))

            meshes = [list(m) for m in self.result['kpoint_mesh']]

            if len(meshes) < 2:
                raise ValueError(f'Not enough valid kpoint_mesh rows for analysis: {len(meshes)}')

            kx = []
            k3 = []
            for m in meshes:
                kx.append(m[0])
                k3.append(m[0]*m[1]*m[2])
            
            k_true=[]
            for i in self.result['uuid']:
                k_true.append(find_pw_parent(load_node(i)).outputs.output_parameters.get_dict()['number_of_k_points'])

            self.result['kx'] = kx
            self.result['k^3'] = k3
            self.result['nk'] = k_true
            
            self.parameters = np.array(meshes)
            self.starting_mesh = self.parameters[0]
            self.p = self.parameters
            self.parameters = self.parameters[:,0]*self.parameters[:,1]*self.parameters[:,2]
            self.delta_z = max(1,self.delta_[2])
            self.delta_y = max(1,self.delta_[1])
            self.delta_x = max(1,self.delta_[0])
            self.delta_ = self.delta_x*self.delta_y*self.delta_z

        else:
            # Extract the parameter grid used in calculations
            self.parameters = np.asarray(self.grid[self.var_[0]])
            # kpoint_density holds the AiiDA distance (1/Å, decreasing with
            # refinement). The plateau/curvature fit machinery is tuned to
            # increasing mesh-count scales (relative meshes ~ 1/d, k-point
            # counts ~ 1/d**3), so fit against the (1/distance)**3 proxy here
            # and map the predicted proxy back to a distance at the end.
            if self.var_[0] == 'kpoint_density' and len(self.parameters) and not np.any(np.asarray(self.parameters) == 0):
                self.parameters = (1.0 / np.asarray(self.parameters, dtype=float))**3
                d0 = getattr(self, 'start', None)
                d1 = getattr(self, 'stop', None)
                if d0 is not None and d1 is not None and d0 != d1:
                    # proxy-space stride with ~8 points per window step
                    self.delta_ = abs(1.0 / d0 - 1.0 / d1)**3 / 8.0

        #self.bb, self.GG = copy.deepcopy(list(self.result.BndsRnXp.values)),copy.deepcopy(list(self.result.NGsBlkXp.values))
        
        self.res = copy.deepcopy(self.result[self.what].values[:] + self.Fermi)
        
        if isinstance(self.bande, (int, float, type(None))):
            self.bb_Ry = 0
        else:
            try:
                self.bb_Ry = copy.deepcopy(self.bande[0,np.array(self.result.BndsRnXp.values)-1]/13.6)
            except (IndexError, TypeError, AttributeError, KeyError): # not a bands convergence
                try:
                    self.bb_Ry = copy.deepcopy(self.bande[0,-1]/13.6)
                except (IndexError, TypeError):
                    self.bb_Ry = 0
            
        self.r[:] = self.r[:] + self.Fermi
        
        #self.G = copy.deepcopy(self.G)
        
        #print('Params',self.parameters)
        
    
            
################################################################

    def fit_space_1D(self,fit=False,alpha=1,beta=1,reference = None,verbose=True,plot=False,dim=100,b=None,g=None,save=False,thr_fx=5e-5):
        
        f = lambda x,a,b: (a/x**alpha + b)
        fx = lambda x,a: -alpha*a/(x**(alpha+1))
         
        xdata,ydata = self.parameters,self.r
        #print('fitting all simulations.')
        if len(ydata) < 2:
            LOGGER.warning(f"Not enough data points ({len(ydata)}) for 1D fit (need >=2), skipping fit")
            return False

        popt,pcov = curve_fit(f,xdata=xdata,
                      ydata=ydata,sigma=1/xdata**(3-alpha), #so we give more importance to high if the fit is slow...c
                      bounds=([-np.inf,-np.inf],[np.inf,np.inf]))
        
        MAE_int = np.average((abs(f(xdata,popt[0],popt[1])-ydata)),weights=xdata)
        LOGGER.debug('MAE fit = {} eV; power law = {}'.format(MAE_int,alpha))
        self.MAE_fit = MAE_int
        
        if verbose: 
            LOGGER.debug(np.max(xdata)*10)
            LOGGER.debug(np.min(xdata))
            LOGGER.debug('conv_thr, %s', self.conv_thr)
            
        ###########Preliminary fit#################################
        
        self.X_fit = np.arange(np.min(xdata), np.max(xdata)*10, abs(self.delta_)) 
        
        if self.var_[0] == 'kpoint_mesh':
            l = [1,2,3]
            for i in range(3):
                if self.delta[i] != 0:
                    if i == 0:
                        l[i] = np.arange(np.min(self.p[:,i]),np.max(self.p[:,i])*10,self.delta[i])
                        length = len(l[i])
                    else:
                        l[i] = l[0]*(np.min(self.p[:,i])/np.min(self.p[:,0]))
                        length = len(l[i])
                else:
                    l[i] = 0 
            for i in range(3):
                if self.delta[i] == 0:
                    l[i] = np.ones(length)
            
            #print('AAAAAAA',length, l)
            self.kx_fit = l[0]
            self.ky_fit = l[1]
            self.kz_fit = l[2]


            l_min = min(len(l[0]),len(l[1]),len(l[2])) #this is needed to match if some kx,ky,kz created with np.arange have different lengths
            self.kx_fit = l[0][:l_min]
            self.ky_fit = l[1][:l_min]
            self.kz_fit = l[2][:l_min]

            self.X_fit = l[0][:l_min]*l[1][:l_min]*l[2][:l_min]
        
        self.Zx_fit = fx(self.X_fit,popt[0])
        
        self.extra = popt[1]

        ###########Estimation of the plateaux corner###############
        
        if reference == 'extra':
            reference = self.extra
        else:
            self.Z_fit = f(self.X_fit,popt[0],popt[1])  
            reference = self.Z_fit[-1]
            
        if self.conv_thr_units=='%':
            thr = self.conv_thr*abs(reference)/100
        else:
            thr = self.conv_thr
            
        if self.var_[0] == 'kpoint_mesh': thr = thr/2
        
        self.condition_conv_calc = np.where(abs(self.Zx_fit)<thr_fx)
        
        if len(self.X_fit[self.condition_conv_calc]) == 0 : return False
        if not b: b = max(max(xdata),self.X_fit[self.condition_conv_calc][0])
            
        #print('b: {}\ng: {}'.format(b,g))
        
        p = f(max(xdata),popt[0],popt[1])
        p_H = f(b,popt[0],popt[1])
        try:
            l = ydata[np.where(xdata == max(xdata))][0]
        except (IndexError, TypeError):
            l = ydata[-1]
        
        LOGGER.debug('extra={} eV, \nlast calculation={} eV \nlast calculation from fit={} eV'.format(round(popt[1],3),
                                                                                          round(l,2),round(p,3)))
        
        if verbose: LOGGER.debug('{} highest point from fit = {} eV\n'.format(b,round(p_H,3)))
        if verbose: LOGGER.debug('\nrelative err extra - last calculation = {}%'.format(round(100*abs((popt[1]-l)/(l)),3)))      
        if verbose: LOGGER.debug('relative err extra - highest point from fit = {}%'.format(round(100*abs((popt[1]-p_H)/(p_H)),3)))
        
        if self.var_[0] == 'kpoint_mesh':
            self.kx_fit = self.kx_fit[self.X_fit <= max(max(xdata),self.X_fit[self.condition_conv_calc][0]*1.5)]
            self.ky_fit = self.ky_fit[self.X_fit <= max(max(xdata),self.X_fit[self.condition_conv_calc][0]*1.5)]
            self.kz_fit = self.kz_fit[self.X_fit <= max(max(xdata),self.X_fit[self.condition_conv_calc][0]*1.5)]
            self.X_fit = self.X_fit[self.X_fit <= max(max(xdata),self.X_fit[self.condition_conv_calc][0]*1.5)]
        else:
            self.X_fit = np.arange(min(xdata), b + 1, abs(self.delta_))
        
    
        
        self.Z_fit = f(self.X_fit,popt[0],popt[1])
        
        self.Zx_fit = fx(self.X_fit,popt[0])
            
        return True
    

    def determine_next_calculation(self,
                                   overconverged_values=None,
                                   plot=False,
                                   reference = None,save=False,):
        
        overconverged_values = [] if overconverged_values is None else overconverged_values
        LOGGER.debug('last point:{} eV'.format(self.Z_fit[-1]))
        
        if reference == 'extra':
            reference = self.extra
        else:
            reference = self.Z_fit[-1]
            
        if self.conv_thr_units=='%':
            thr = self.conv_thr*abs(reference)/100
        else:
            thr = self.conv_thr
        
        if self.var_[0] == 'kpoint_mesh': thr = thr/2

        
        #print(thr)
        d = 1%thr
        if d == 0: d = 1
        discrepancy = np.round(abs(reference-self.Z_fit),abs(int(np.round(np.log10(d),0))))
        condition = np.where((discrepancy<=thr))
        
        #print(condition)
        #print(self.Z_fit[condition])
        #print('\n')
        
        #print('Min G condition')
        #print(self.Z_fit[condition][0])
        #print(self.X_fit[condition][0])
        
        self.condition = condition
         
        conv_bands = self.X_fit[condition][0]
        conv_z = self.Z_fit[condition][0]
        
        self.next_step = {
            self.var_[0]:conv_bands,
            self.what: conv_z,
            'already_computed': False,
        }
        
        if conv_bands in self.parameters:
            self.next_step['already_computed'] = True

        if self.var_[0] == 'kpoint_density':
            # fit ran on the (1/distance)**3 proxy: map the converged proxy
            # back to the AiiDA distance carried by the variable.
            self.next_step[self.var_[0]] = 1.0 / max(conv_bands, np.finfo(float).eps)**(1.0 / 3.0)

        if 'BndsRnXp' in self.var and 'GbndRnge' in self.var:
            self.next_step['GbndRnge'] = copy.deepcopy(self.next_step['BndsRnXp'])
        
        if self.var_[0] == 'kpoint_mesh':
            matching = np.where(self.X_fit == conv_bands)[0]
            if len(matching) == 0:
                matching = np.array([int(np.argmin(np.abs(self.X_fit - conv_bands)))])
            kx = float(self.kx_fit[matching[0]])
            A = self.starting_mesh[1]/self.starting_mesh[0]
            B = self.starting_mesh[2]/self.starting_mesh[0]
            factor = (kx-self.starting_mesh[0])/self.delta[0]
            ky = factor*self.delta[1] + self.starting_mesh[1]
            kz = factor*self.delta[2] + self.starting_mesh[2]
            
            #in order to get compatible kx,ky,kz with respect to the starting ones.
            ky = int(kx*A + kx*A%1)
            kz = int(kx*B + kx*B%1)

            if self.delta[0] == 0: kx = 1
            if self.delta[1] == 0: ky = 1            
            if self.delta[2] == 0: kz = 1

            self.next_step[self.var_[0]] = [int(kx),int(ky),int(kz)]
        
            for k in range(3):
                if self.next_step[self.var_[0]][k] > self.max[k]:
                    self.next_step['new_grid'] = True
                    break
        else:
            if self.var_[0] == 'kpoint_density':
                # Decreasing variable: refinement means smaller distances, so the
                # out-of-grid boundary is on the fine side. A next step finer
                # than the current window's fine edge (stop) needs a new window.
                stop = getattr(self, 'stop', None)
                if stop is not None and self.next_step[self.var_[0]] < stop:
                    self.next_step['new_grid'] = True
            elif self.next_step[self.var_[0]] > self.max:
                self.next_step['new_grid'] = True
        
        self.next_step['E_ref'] = reference
        LOGGER.debug('guessed next step: {} \n\n\n'.format(self.next_step))

        return self.next_step
    
    def check_the_point(self,old_hints=None):
        # Guard against missing keys after FIFO reset or first grid (old_hints may be just {'new_grid': True})
        old_hints = {} if old_hints is None else old_hints
        if self.what not in old_hints or self.var_[0] not in old_hints:
            self.point_reached = False
            self.old_discrepancy = np.inf
            self.index = [0]
            return None
        
        if self.var_[0] == 'kpoint_mesh':
            self.old_discrepancy = \
            abs(old_hints[self.what] - self.result[(self.result['kx']==old_hints['kpoint_mesh'][0])][self.what].values[0])
            self.index = [int(self.result[(self.result['kx']==old_hints['kpoint_mesh'][0])].index.values[0])]

            explicit_gw_result=self.result[(self.result['kx']==old_hints['kpoint_mesh'][0])][self.what].values[0]
            LOGGER.debug('Discrepancy with old prediction: {} eV'.format(self.old_discrepancy))

            if old_hints['kpoint_mesh'][0] == self.next_step[self.var_[0]][0]:
                self.point_reached = True
            else:
                self.point_reached = False
                LOGGER.debug('new point predicted.')
            
        else:
            self.old_discrepancy = \
            abs(old_hints[self.what] - self.result[(self.result[self.var_[0]]==old_hints[self.var_[0]])][self.what].values[0])
            self.index = [int(self.result[(self.result[self.var_[0]]==old_hints[self.var_[0]])].index.values[0])]
            
            explicit_gw_result=self.result[(self.result[self.var_[0]]==old_hints[self.var_[0]])][self.what].values[0]
            LOGGER.debug('Discrepancy with old prediction: {} eV'.format(self.old_discrepancy))

            if old_hints[self.var_[0]] == self.next_step[self.var_[0]]:
                self.point_reached = True
            else:
                self.point_reached = False
                LOGGER.debug('new point predicted.')
              
        return explicit_gw_result
    
    
    def analyse(self,old_hints=None,reference = None, plot= False,save_fit=False,save_next = False,colormap='viridis',thr_fx=5e-5):
        
        old_hints = {} if old_hints is None else old_hints
        self.check_passed = True
        error = 10
        power_laws = [1,2,3]
        
        if 'kpoint_mesh' in self.var_: power_laws = [1,2]

        ii = 1
        for i in power_laws:
                self.check_passed = self.fit_space_1D(fit=True,alpha=i,beta=1,plot=False,dim=10,thr_fx=thr_fx)
                if self.MAE_fit<error: 
                    ii = i
                    error = self.MAE_fit
                #THERE SHoULD BE SoME ERRoR FLAG.
                
                
                

        LOGGER.debug('Best power law: {}'.format(ii))  
        
        self.check_passed = self.fit_space_1D(fit=True,alpha=ii,beta=1,verbose=True,plot=plot,save=save_fit,thr_fx=thr_fx)
        
        if not self.check_passed:
            self.point_reached = False
            self.next_step = {'new_grid':True}
            return
        else:
            self.determine_next_calculation(plot=plot, save=save_next,reference=reference)
        
        if 'new_grid' in self.next_step.keys():
            if self.next_step['new_grid']: 
                self.check_passed = False
                self.point_reached = False
                return
            
        self.point_reached = False
        
        if reference == 'extra':
            reference = self.extra
        else:
            reference = self.Z_fit[-1]
            
        if self.conv_thr_units=='%':
            factor = 100/abs(reference)
        else:
            factor=1

        if old_hints or self.next_step['already_computed']:
                if self.next_step['already_computed']: 
                    old_hints = self.next_step
                    self.point_reached = True
                    
                converged_value = self.check_the_point(old_hints)

                if reference == 'extra':
                    reference = self.extra
                else:
                    reference = self.Z_fit[-1]
                
                d = 1%(self.conv_thr/factor)
                if d == 0 : d = 1
                if np.round(abs(self.old_discrepancy),abs(int(np.round(np.log10(d),0)))) <= self.conv_thr/factor: 
                    self.check_passed = True
                else:
                    self.check_passed = False

        if not self.check_passed and self.point_reached:
            self.next_step['new_grid'] = True
        elif self.MAE_fit > 10*self.conv_thr*factor:
            self.next_step['new_grid'] = True
        else:
            self.next_step['new_grid'] = False
        
        #converged:
        if self.check_passed and self.point_reached: #set the value as the explicit GW computed one
            self.next_step[self.what] = converged_value

        return True