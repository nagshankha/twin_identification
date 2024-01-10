import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz

class ReadOrientationFile:
   
   def __init__(self, filepath, image_flags_included = True):
      
      self.filepath = filepath
      self.get_num_grains_n_box_lengths()
      self.read_grain_orientations()
      self.read_atoms(image_flags_included)
      
   def get_num_grains_n_box_lengths(self):
      
      if self.filepath[-3:] == '.gz':
         import gzip
         f = gzip.open(self.filepath, 'rt')
      else:
         f = open(self.filepath, 'rt')
      
      for i, line in enumerate(f):
         if i == 1:
            self.num_grains = int(line.split()[-1])
         elif i == 2:
            self.box_dimensions = list(map(float, line.split()[-6:]))
            
      f.close()
         
   def read_grain_orientations(self):
      
      dtype = [('GrainID', int), ('GrainOrientation.q1', float),
               ('GrainOrientation.q2', float), ('GrainOrientation.q3', float),
               ('GrainOrientation.q4', float)]
      
      self.grain_id_orient = np.loadtxt(self.filepath, skiprows=5, 
                                        max_rows=self.num_grains,
                                        dtype=dtype)
      
   def read_atoms(self, image_flags_included):
      
      if image_flags_included:
         dtype = [('ParticleIdentifier', int), ('ParticleType', int), ('StructureType', int),
                  ('Grain', int), ('Position.x', float), ('Position.y', float),
                  ('Position.z', float), ('Image.x', int), ('Image.y', int),
                  ('Image.z', int)]
      else:
         dtype = [('ParticleIdentifier', int), ('ParticleType', int), ('StructureType', int),
                  ('Grain', int), ('Position.x', float), ('Position.y', float),
                  ('Position.z', float)]
      
      self.atom_info = np.loadtxt(self.filepath, skiprows=self.num_grains+7,
                                        dtype=dtype)
      
      
class FindGrainCorrespondence:
   
   def __init__(self, grain_orient_initial_config, grain_orient_current_config):
      
      self.grain_orient_initial_config = grain_orient_initial_config
      self.grain_orient_current_config = grain_orient_current_config
      
   def check_grains_with_mixed_structure_types(self):
      
      arr1 = []; arr2 = []
      
      fcc_grains_initial_config = []
      hcp_grains_initial_config = []
      fcc_grains_current_config = []
      hcp_grains_current_config = []
      
      for i in np.arange(self.grain_orient_initial_config.num_grains)+1:
         u = np.unique(self.grain_orient_initial_config.atom_info['StructureType'][
                                         self.grain_orient_initial_config.atom_info['Grain']==i])
         if len(u) > 1:
            n_grain_atoms = np.count_nonzero(self.grain_orient_initial_config.atom_info['Grain']==i)
            n_other = np.count_nonzero(self.grain_orient_initial_config.atom_info['StructureType'][
                                  self.grain_orient_initial_config.atom_info['Grain']==i] == 0)
            n_fcc = np.count_nonzero(self.grain_orient_initial_config.atom_info['StructureType'][
                                  self.grain_orient_initial_config.atom_info['Grain']==i] == 1)
            n_hcp = np.count_nonzero(self.grain_orient_initial_config.atom_info['StructureType'][
                                  self.grain_orient_initial_config.atom_info['Grain']==i] == 2)
            
            arr1.append((i, n_grain_atoms, n_fcc, n_hcp, n_other))
         elif u[0] == 1:
            fcc_grains_initial_config.extend([i])
         elif u[0] == 2:
            hcp_grains_initial_config.extend([i])
            
      for i in np.arange(self.grain_orient_current_config.num_grains)+1:
         u = np.unique(self.grain_orient_current_config.atom_info['StructureType'][
                                         self.grain_orient_current_config.atom_info['Grain']==i])
         if len(u) > 1:
            n_grain_atoms = np.count_nonzero(self.grain_orient_current_config.atom_info['Grain']==i)
            n_other = np.count_nonzero(self.grain_orient_current_config.atom_info['StructureType'][
                                  self.grain_orient_current_config.atom_info['Grain']==i] == 0)
            n_fcc = np.count_nonzero(self.grain_orient_current_config.atom_info['StructureType'][
                                  self.grain_orient_current_config.atom_info['Grain']==i] == 1)
            n_hcp = np.count_nonzero(self.grain_orient_current_config.atom_info['StructureType'][
                                  self.grain_orient_current_config.atom_info['Grain']==i] == 2)
            
            arr2.append((i, n_grain_atoms, n_fcc, n_hcp, n_other))
         elif u[0] == 1:
            fcc_grains_current_config.extend([i])
         elif u[0] == 2:
            hcp_grains_current_config.extend([i])
            
      self.fcc_grains_initial_config = np.array(fcc_grains_initial_config)
      self.hcp_grains_initial_config = np.array(hcp_grains_initial_config)
      self.fcc_grains_current_config = np.array(fcc_grains_current_config)
      self.hcp_grains_current_config = np.array(hcp_grains_current_config)
            
      dtype = [('GrainID', int), ('natoms', int), ('n_fcc', int), ('n_hcp', int), 
               ('n_other', int)]      
      self.mixed_grains_initial_config = np.array(arr1, dtype=dtype)
      self.mixed_grains_current_config = np.array(arr2, dtype=dtype)
      
   def find_correspondence(self, grain_corr_cutoff = 0.0):
      
      row_ind = []; col_ind = []; data = []
      for i in range(self.grain_orient_current_config.num_grains):
         n_atoms = 0
         for j in range(self.grain_orient_initial_config.num_grains+1):
            u = np.in1d(self.grain_orient_current_config.atom_info['ParticleIdentifier'][
                                       self.grain_orient_current_config.atom_info['Grain']==i+1],
                        self.grain_orient_initial_config.atom_info['ParticleIdentifier'][
                                       self.grain_orient_initial_config.atom_info['Grain']==j])
            if np.count_nonzero(u) != 0:
               n_atoms += np.count_nonzero(u)
               row_ind.extend([i]); col_ind.extend([j])
               data.extend([np.count_nonzero(u)])
               
            if n_atoms == np.count_nonzero(self.grain_orient_current_config.atom_info['Grain']==i+1):
               break
         
         if n_atoms != np.count_nonzero(self.grain_orient_current_config.atom_info['Grain']==i+1):
            raise RuntimeError('Some atoms in the current configuration')
               
      self.atom_redistribution = csr_matrix((data, (row_ind, col_ind)),
                                             shape=(self.grain_orient_current_config.num_grains, 
                                                    self.grain_orient_initial_config.num_grains+1))
      
      u = self.atom_redistribution/np.sum(self.atom_redistribution, axis=1)
      init_grain_id = np.argmax(u[:,1:], axis=1)+1
      max_value = np.max(u[:,1:], axis=1)
      current_grain_id = np.nonzero(max_value>grain_corr_cutoff)[0]+1
      init_grain_id = init_grain_id[current_grain_id-1]
      max_value = max_value[current_grain_id-1]
      self.grain_correspondence = np.array(list(zip(current_grain_id, 
                                                    init_grain_id, 
                                                    max_value)),
                     dtype=[('CurrentGrainID', int), ('InitialGrainID', int),
                            ('FractionAtomsRetained', float)] )
      
      
      
   def save(self, folder):
      
      if hasattr(self, 'fcc_grains_initial_config'):
         
         np.save(folder+'fcc_grains_initial_config.npy', 
                 self.fcc_grains_initial_config)
         np.save(folder+'hcp_grains_initial_config.npy', 
                 self.hcp_grains_initial_config)
         np.save(folder+'fcc_grains_current_config.npy', 
                 self.fcc_grains_current_config)
         np.save(folder+'hcp_grains_current_config.npy', 
                 self.hcp_grains_current_config)
         
         np.save(folder+'mixed_grains_initial_config.npy',
                 self.mixed_grains_initial_config)
         np.save(folder+'mixed_grains_current_config.npy',
                 self.mixed_grains_current_config)
         
      if hasattr(self, 'grain_correspondence'):
         
         save_npz(folder+'atom_redistribution.npz', self.atom_redistribution)
         np.save(folder+'grain_correspondence.npy', self.grain_correspondence)       
         
      
   def read(self, folder):
      
      self.fcc_grains_initial_config = np.load(folder+
                                               'fcc_grains_initial_config.npy')
      self.hcp_grains_initial_config = np.load(folder+
                                               'hcp_grains_initial_config.npy')
      self.fcc_grains_current_config = np.load(folder+
                                               'fcc_grains_current_config.npy')
      self.hcp_grains_current_config = np.load(folder+
                                               'hcp_grains_current_config.npy')
      
      self.mixed_grains_initial_config = np.load(folder+
                                                 'mixed_grains_initial_config.npy')
      self.mixed_grains_current_config = np.load(folder+
                                                 'mixed_grains_current_config.npy')
      
      self.atom_redistribution = load_npz(folder+'atom_redistribution.npz')
      self.grain_correspondence = np.load(folder+'grain_correspondence.npy')
      
      
class CalculateDisorientationAngles:
   
   def __init__(self, grain_corr_obj):
      
      self.grain_corr_obj = grain_corr_obj
      
   def calc_disorientation(self):
      
      from ovito.modifiers import PolyhedralTemplateMatchingModifier
      
      disorientation_angles = []
      
      for i in range(len(self.grain_corr_obj.grain_correspondence)):
         
         if (self.grain_corr_obj.grain_correspondence['CurrentGrainID'][i] not in 
             self.grain_corr_obj.fcc_grains_current_config):
            continue
         
         current_grain_num = self.grain_corr_obj.grain_correspondence[
                                                         'CurrentGrainID'][i]
         initial_grain_num = self.grain_corr_obj.grain_correspondence[
                                                         'InitialGrainID'][i]
         
         
         disorientation_angles.append(( current_grain_num,
            np.rad2deg(PolyhedralTemplateMatchingModifier.calculate_misorientation(
            self.grain_corr_obj.grain_orient_current_config.grain_id_orient[
               self.grain_corr_obj.grain_orient_current_config.grain_id_orient[
                                      'GrainID']==current_grain_num][
               ['GrainOrientation.q1', 'GrainOrientation.q2', 
                'GrainOrientation.q3', 'GrainOrientation.q4']].tolist(), 
            self.grain_corr_obj.grain_orient_initial_config.grain_id_orient[
               self.grain_corr_obj.grain_orient_initial_config.grain_id_orient[
                                      'GrainID']==initial_grain_num][
               ['GrainOrientation.q1', 'GrainOrientation.q2', 
                'GrainOrientation.q3', 'GrainOrientation.q4']].tolist(), 
            symmetry='cubic') ) ) )
         
      self.disorientation_angles = np.array(disorientation_angles,
                                            dtype=[('GrainID', int),
                                                    ('Disorientation Angle', 
                                                     float)])
      
   def save(self, folder):
      
      np.save(folder+'disorientation_angles.npy', self.disorientation_angles)
      
   def dump_file_with_orientation_data(self, filepath, image_flags_included=True):
      
      natoms = len(self.grain_corr_obj.grain_orient_current_config.atom_info)
      box_dims = self.grain_corr_obj.grain_orient_current_config.box_dimensions
      
      disorientation_angles2 = -1*np.ones(natoms)
      for i,g in enumerate(self.disorientation_angles['GrainID']):
         disorientation_angles2[self.grain_corr_obj.grain_orient_current_config.atom_info[
            'Grain']==g] = self.disorientation_angles['Disorientation Angle'][i]
      
      import io_files
      import numpy.lib.recfunctions as rfn
      
      if image_flags_included:
         dump_data = rfn.merge_arrays([self.grain_corr_obj.grain_orient_current_config.atom_info[
                                       ['ParticleIdentifier', 'ParticleType', 'Position.x', 
                                        'Position.y', 'Position.z', 'Image.x', 'Image.y',
                                        'Image.z', 'StructureType', 'Grain']],
                                       np.array(disorientation_angles2[:,None], 
                                                dtype = [('DisorientationAngles', float)])], 
                                      flatten = True, usemask = False)
      else:
         dump_data = rfn.merge_arrays([self.grain_corr_obj.grain_orient_current_config.atom_info[
                                       ['ParticleIdentifier', 'ParticleType', 'Position.x', 
                                        'Position.y', 'Position.z', 'StructureType', 'Grain']],
                                       np.array(disorientation_angles2[:,None], 
                                                dtype = [('DisorientationAngles', float)])], 
                                      flatten = True, usemask = False)
      
      io_files.write_lammps_dump_file(filepath, 1, natoms, ['pp', 'pp', 'pp'], 
                                 box_dims[:2], box_dims[2:4], box_dims[4:], 
                                 dump_data)
         
      
      
      
