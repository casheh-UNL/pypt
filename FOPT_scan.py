# parameter scan for first-order phase transitions

from .constants import *
from .pt_math import *

from .ftpot import *


class FOPTScan(object):
    def __init__(self, alpha_BL0_range=[0, 0.25], alpha_Y0_range=[0, 0.25], mu=1, scan_points=1e2, find_points=1, log10_dist=False):
        self.alpha_BL0_min, self.alpha_BL0_max = alpha_BL0_range
        self.alpha_Y0_min, self.alpha_Y0_max = alpha_Y0_range
        self.mu = mu

        self.scan_points = int(scan_points)
        self.find_points = int(find_points)
        self.log10_dist = log10_dist

        self.results = []
        self.failed_points = []
    
    def scan(self):
        if self.log10_dist:
            alpha_BL0 = 10**( np.random.uniform(np.log10(self.alpha_BL0_min), np.log10(self.alpha_BL0_max)) )
            alpha_Y0  = 10**( np.random.uniform(np.log10(self.alpha_Y0_min),  np.log10(self.alpha_Y0_max)) )
        else:
            alpha_BL0 = np.random.uniform(self.alpha_BL0_min, self.alpha_BL0_max)
            alpha_Y0  = np.random.uniform(self.alpha_Y0_min,  self.alpha_Y0_max)

        result = VeffBL(alpha_BL0=alpha_BL0, alpha_Y0=alpha_Y0, mu=self.mu).find_critical_phi_and_temperature() # returns np.array([phi_crit, T_crit])
        # result = ftpot_CW.VeffBL(g=alpha_BL0, y=alpha_Y0, mu=self.mu).find_critical_phi_and_temperature() # returns np.array([phi_crit, T_crit])
        
        if result is not None:
            result_entry = {'alpha_BL0':alpha_BL0, 'alpha_Y0':alpha_Y0, 'mu':self.mu, 'phi_crit':result[0], 'T_crit':result[1]}
            self.results.append(result_entry)
        
        else:
            failed_point = {'alpha_BL0':alpha_BL0, 'alpha_Y0':alpha_Y0, 'mu':self.mu}
            self.failed_points.append(failed_point)

    def perform_scan(self):
        for _ in range(self.scan_points):
            self.scan()

    def perform_find_scan(self):
        while len(self.results) < self.find_points:
            self.scan()

    def plot_heatmap_scatter(self, save_dir=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        # from matplotlib import rcParams
        # rcParams['text.usetex'] = True

        alpha_BL0_list = []
        alpha_Y0_list = []
        PT_strength_list = []
        failed_alpha_BL0_list = []
        failed_alpha_Y0_list = []

        for j in self.results:
            alpha_BL0, alpha_Y0, mu, phi_crit, T_crit = j.values()

            alpha_BL0_list.append(alpha_BL0)
            alpha_Y0_list.append(alpha_Y0)
            PT_strength_list.append(phi_crit / T_crit)
        
        for j in self.failed_points:
            alpha_BL0, alpha_Y0, mu = j.values()
            failed_alpha_BL0_list.append(alpha_BL0)
            failed_alpha_Y0_list.append(alpha_Y0)

        plt.scatter(failed_alpha_BL0_list, failed_alpha_Y0_list, color='lightgray')

        plt.scatter(alpha_BL0_list, alpha_Y0_list, c=PT_strength_list, cmap='viridis', norm=LogNorm())
        plt.colorbar(label=r'$\phi_c / T_c$')
        plt.xlabel(r'$\alpha_{B-L}(0)$')
        plt.ylabel(r'$\alpha_{Y_i}(0)$')
        plt.title(f'FOPTs: {len(self.results)}')
        if self.log10_dist:
            plt.xscale('log')
            plt.yscale('log')
            
        if save_dir:
            plt.savefig(save_dir+'FOPT_scan.png', dpi=300)

        plt.show()

    def __call__(self):
        pass