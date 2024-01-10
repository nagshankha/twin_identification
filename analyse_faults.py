import glob, re
import numpy as np
import io_files
from ovito.io import *
from ovito.pipeline import StaticSource, Pipeline
from ovito.data import *
from ovito.modifiers import *


class FaultAnalysis:
   
   def __init__(self, dumpfilepath_pattern, output_folder, file_num_index=None):
      
      dump_files = glob.glob(dumpfilepath_pattern)
      file_nums = []
      for f in dump_files:
         n = re.findall('\d+', f.split('/')[-1])
         if len(n) == 0:
            raise RuntimeError('Cannot screen pattern')
         elif len(n) == 1:
            file_nums.extend([int(n[0])])
         else:
            file_nums.extend([int(n[file_num_index])])
            
      self.dump_files = np.array(dump_files)[np.argsort(file_nums)] 
      self.file_nums = np.sort(file_nums)
      self.output_folder = output_folder
      
   def read_dump(self, file_path):
      
      self.n_atoms_1, self.atom_info_1 = np.array(io_files.read_data_from_LAMMPS_dump_file(
                                               file_path, 
                                               {'id': ('id', 'i8'),
                                                'StructureType': ('StructureType', 'i4'),
                                                'Grain': ('Grain', 'i8'),
                                                'DisorientationAngles': (
                                                   'DisorientationAngles', 'f8') }),
                                             dtype=object)[[1,-1]]
            
   def select_fcc_hcp_gb_twin_atoms(self, i):
      
      self.sf_atoms_1 = np.logical_and( self.atom_info_1['Grain'] > 0,
                                        self.atom_info_1['StructureType']==2 )
      self.twin_atoms_1 = np.argmin(
                     abs(np.c_[self.atom_info_1['DisorientationAngles'], 
                           70.53 - self.atom_info_1['DisorientationAngles']]),
                     axis = 1 ).astype(bool)
      self.n_sf_atoms_1 = np.count_nonzero(self.sf_atoms_1)
      self.n_twin_atoms_1 = np.count_nonzero(self.twin_atoms_1)
      self.n_fcc_atoms = np.count_nonzero(np.logical_and( 
                                        self.atom_info_1['Grain'] > 0,
                                        self.atom_info_1['StructureType']==1 ))
      self.n_gb_atoms = np.count_nonzero(self.atom_info_1['Grain'] == 0)
      
      if (self.n_fcc_atoms+self.n_gb_atoms+self.n_sf_atoms_1) != self.n_atoms_1:
         sel_atoms = np.logical_and(self.atom_info_1['Grain'] > 0,
                                    np.in1d(self.atom_info_1['StructureType'], [1,2],
                                            invert=True))
         reduced_arr = self.atom_info_1[sel_atoms][['id', 'StructureType', 'Grain']]
         np.savetxt(self.output_folder+f'non_fcc_non_hcp_non_gb_atoms_after_{(i+1)*10}_ps.txt', 
                    reduced_arr, fmt='%d', 
                    header=(f'n_atoms = {np.count_nonzero(sel_atoms)} '+
                            f'({np.count_nonzero(sel_atoms)/self.n_atoms_1*100} %) \n'+
                            ' '.join(reduced_arr.dtype.names)))
         
      
   def hcp_atoms_created_destroyed(self):
      
      self.sf_atoms_created = np.count_nonzero(
                              np.in1d(self.atom_info_1['id'][self.sf_atoms_1],
                                      self.atom_info_0['id'][self.sf_atoms_0],
                                      invert=True))
         
      self.sf_atoms_destroyed = np.count_nonzero(
                                 np.in1d(self.atom_info_0['id'][self.sf_atoms_0],
                                         self.atom_info_1['id'][self.sf_atoms_1],
                                         invert=True))
      
      self.twin_atoms_created = np.count_nonzero(
                              np.in1d(self.atom_info_1['id'][self.twin_atoms_1],
                                      self.atom_info_0['id'][self.twin_atoms_0],
                                      invert=True))
         
      self.twin_atoms_destroyed = np.count_nonzero(
                                 np.in1d(self.atom_info_0['id'][self.twin_atoms_0],
                                         self.atom_info_1['id'][self.twin_atoms_1],
                                         invert=True))
      
   def analyse(self):
      
      f = open(self.output_folder+'hcp_twin_evolution.txt', 'wt', buffering=1)
      f.write('# Time(ps) z-Strain n_atoms percent_fcc_atoms percent_gb_atoms '+
              'percent_sf_atoms percent_twin_atoms '+
              'num_sf_atoms_created num_sf_atoms_destroyed '+
              'num_twin_atoms_created num_twin_atoms_destroyed \n')

      for i, df in enumerate(self.dump_files):
      
         self.read_dump(df)
         self.select_fcc_hcp_gb_twin_atoms(i)
         
         if i == 0:
            self.sf_atoms_created = 0
            self.sf_atoms_destroyed = 0
            self.twin_atoms_created = 0
            self.twin_atoms_destroyed = 0
         else:
            self.hcp_atoms_created_destroyed()
         
         f.write(' '.join(map(str, [(i+1)*10, (i+1)*1e-3, self.n_atoms_1,
                                    (self.n_fcc_atoms/self.n_atoms_1*100),
                                    (self.n_gb_atoms/self.n_atoms_1*100),
                                    (self.n_sf_atoms_1/self.n_atoms_1*100),
                                    (self.n_twin_atoms_1/self.n_atoms_1*100),
                                    self.sf_atoms_created,
                                    self.sf_atoms_destroyed,
                                    self.twin_atoms_created,
                                    self.twin_atoms_destroyed])) + '\n')
         
         self.atom_info_0 = self.atom_info_1.copy()
         self.sf_atoms_0 = self.sf_atoms_1.copy()
         self.twin_atoms_0 = self.twin_atoms_1.copy()
         self.n_sf_atoms_0 = self.n_sf_atoms_1
         self.n_twin_atoms_0 = self.n_twin_atoms_1
         
         
      f.close()
      
      
class FaultAspectRatioAnalysis(FaultAnalysis):
   
   def __init__(self, dumpfilepath_pattern, file_num_index=None,
                twin_disorient_angle_lo = 40):
      
      FaultAnalysis.__init__(self, dumpfilepath_pattern, file_num_index)
      
      self.twin_disorient_angle_lo = twin_disorient_angle_lo
      
   def __initiate_ovito_pipeline__(self, structure_file_path):
      
      # Load the simulation dataset to be analyzed.
      self.pipeline = import_file(structure_file_path)
      
   def __loop_over_sfs_n_twins__(self):
      
      self.pipeline.append(ExpressionSelectionModifier(
         expression = (f'(disorientationangles > self.twin_disorient_angle_lo) '+
                       '|| (grain > 0 && structuretype == 2)')))
      data = self.pipeline.compute()
      grain_id_list = np.unique(data.particles['Grain'][
                                   data.particles['Selection']])
      del self.pipeline[-1]
      data = self.pipeline.compute()
      
      self.aspect_ratio_info = []
      for grain_id in grain_id_list:
         self.__analyze__(grain_id)
      
   def __analyze__(self, grain_id):
      
      data = self.pipeline.compute()
      
      self.pipeline.append(ExpressionSelectionModifier(
         expression = f'grain != grain_id'))
         
      self.pipeline.append(DeleteSelectedModifier())
      
      self.pipeline.append(ConstructSurfaceModifier(
         identify_regions = True, map_particles_to_regions = True,
         select_surface_particles=True))
      
      data = self.pipeline.compute()
      
      # mind it: this is a common case
      #if np.count_nonzero(data.surfaces['surface'].regions['Filled']) > 1:
      #   raise RuntimeError(f'Twinned grainID = {grain_id}: Two regions in a twin. '+
      #                      'Maybe a issue with grain segmentation')
      
      n_filled_regions = np.count_nonzero(data.surfaces['surface'].regions['Filled'])
          
      n_atoms_inside_region = np.count_nonzero(
         data.surfaces['surface'].regions['Filled'][
            data.particles['Region']])
      n_atoms_on_surface = np.count_nonzero(data.particles['Selection'])
      
      
      
      
      
         
      
      
      original_cell = np.array(data.cell)
         
      # Polyhedral Template Matching
      pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(
                                rmsd_cutoff = rmsd_cutoff,
                                output_orientation=True))
      
      # Affine Transformation to initial cell
      if target_cell is not None:
         pipeline.modifiers.append(AffineTransformationModifier(
                                    relative_mode = False,
                                    target_cell = target_cell))
      
      # Grain Segmentation
      pipeline.modifiers.append(GrainSegmentationModifier(
                                color_particles=False,
                                handle_stacking_faults=False,
                                orphan_adoption=False))
      
      # Affine Transformation to original cell
      if target_cell is not None:
         pipeline.modifiers.append(AffineTransformationModifier(
                                    relative_mode = False,
                                    target_cell = original_cell))
      
   
      
   
      
    
   
      
   
      

         
         
      