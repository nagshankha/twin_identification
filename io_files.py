##############################################
######## IMPORT PACKAGES #####################
import numpy as np
from lammps import lammps
import os
import xml.etree.ElementTree as ET
import pickle
##############################################
##############################################


#######################################################################
############## ROUTINES AND SUB-ROUTINES ##############################
#######################################################################

def read_lammps_data_file(datafile_path, gather_atom_inputs, **kwargs):

   """
   Reads a LAMMPS data file with the python API of LAMMPS

   INPUTS: 
   datafile_path (type: string): path to the LAAMPS datafile to be parsed
   gather_atom_inputs (type: dict): gather atom inputs in the following format ::
                                    {input_name1: [name,type,count], input_name2: [name,type,count] ...}
                                     --> consult LAMMPS python interface documentation for the command gather_atoms

   Optional named inputs for force computation:
   boundary (type: string): consult LAMMPS documentation
   potential_file_path (type: string)
   pair_coeff_atoms (type: list of strings): atoms in pair_coeff command (consult LAMMPS documentation) 

   OUTPUTS:
   type: dict -> {input_name1: ndarray, input_name2: ndarray}

   """

   lmp = lammps(cmdargs = ['-log', 'none'])
   lmp.command('boundary p p p')
   lmp.command('read_data {0}'.format(datafile_path))
   res = {}
   res.update(box_dims = np.array([lmp.extract_global('boxxlo', 1), lmp.extract_global('boxxhi', 1), 
                                   lmp.extract_global('boxylo', 1), lmp.extract_global('boxyhi', 1),
                                   lmp.extract_global('boxzlo', 1), lmp.extract_global('boxzhi', 1)]))
   lmp.close()

   lmp = lammps(cmdargs = ['-log', 'none'])

   if kwargs == {}:
      if any(np.array(map(lambda x: x[0], gather_atom_inputs.values())) == 'f'):
         lmp.close()
         raise RuntimeError('For force computation, "boundary", "potential_file_path" and "pair_coeff_atoms" is needed')
      lmp.command('boundary s s s')
      lmp.command('read_data {0}'.format(datafile_path))
      
      for key in gather_atom_inputs.keys():
         param = gather_atom_inputs[key]
         if param[2] == 1:
            res.update({key: np.array(lmp.gather_atoms(param[0], param[1], param[2]))})
         else:
            res.update({key: np.array(lmp.gather_atoms(param[0], param[1], param[2])).reshape(-1, param[2])})
   else:
      if set(kwargs.keys()) != set(['boundary', 'potential_file_path', 'pair_coeff_atoms']):
         lmp.close()
         raise RuntimeError('Either all three or no optional arguments viz. "boundary", "potential_file_path" and "pair_coeff_atoms" must be provided')

      lmp.command('units metal')
      lmp.command('atom_style atomic')
      lmp.command('boundary {0}'.format(kwargs['boundary']))
      lmp.command('read_data {0}'.format(datafile_path))
      lmp.command('pair_style eam/alloy')
      lmp.command('pair_coeff * * {0} {1}'.format(kwargs['potential_file_path'], ' '.join(kwargs['pair_coeff_atoms'])))
      lmp.command('run 0')
      
      for key in gather_atom_inputs.keys():
         param = gather_atom_inputs[key]
         if param[2] == 1:
            res.update({key: np.array(lmp.gather_atoms(param[0], param[1], param[2]))})
         else:
            res.update({key: np.array(lmp.gather_atoms(param[0], param[1], param[2])).reshape(-1, param[2])})

   lmp.close()

   return res

##############################################################################################################

def write_lammps_data_file(file_path, n_types, atom_types:int or np.ndarray, 
                           xyz, box_dims, image_flags = None, ids = None, 
                           masses = None, velocities = None, tilt_values = None,
                           file_description = None):
   
   if (isinstance(xyz, np.ndarray) and len(np.shape(xyz)) == 2 and 
       np.shape(xyz)[1] == 3 and np.issubdtype(xyz.dtype, np.floating)):
      n_atoms = np.shape(xyz)[0]
   else:
      raise ValueError('Atom positions array "xyz" must be a floating numpy '+
                       '2D array with 3 columns, each column corresponding to '+
                       'a spatial dimension in 3D space')

   if ids is None:
      ids = np.arange(n_atoms) + 1
   elif not (isinstance(ids, np.ndarray) and len(np.shape(ids)) == 1 and 
       len(ids) == n_atoms and np.issubdtype(ids.dtype, np.integer)):
      raise ValueError('Atom IDs array "ids" must be an integer numpy '+
                       '1D array of length equal to the number of atoms')
      
   if not (isinstance(n_types, int) and n_types > 0):
      raise ValueError('"n_types" must be a positive integer specifying the maximum number '+
                       'of atom types in the system')
   

   if (isinstance(atom_types, int) and atom_types > 0 and atom_types <= n_types):
      atom_types = atom_types * np.ones(n_atoms, dtype = np.int)
   elif not (isinstance(atom_types, np.ndarray) and len(np.shape(atom_types)) == 1 and 
       len(atom_types) == n_atoms and np.issubdtype(atom_types.dtype, np.integer)
       and np.max(atom_types) <= n_types and np.min(atom_types) > 0):
      raise ValueError('"atom_types" must be a positive integer numpy 1D array of atom '+
                       'types, of length equal to the number of atoms, or a '+
                       'positive integer if all the atoms are of the same type.')

   if image_flags is not None:
      if not (isinstance(image_flags, np.ndarray) and len(np.shape(image_flags)) == 2 and 
          np.shape(image_flags)[1] == 3 and len(image_flags) == n_atoms and 
          np.issubdtype(image_flags.dtype, np.integer)):
         raise ValueError('Image flags array "image_flags" must be an integer numpy '+
                          '2D array of length equal to the number of atoms, '+
                          'with 3 columns, each column corresponding to '+
                          'a spatial dimension in 3D space') 
      
   if not (isinstance(box_dims, np.ndarray) and len(np.shape(box_dims)) == 1 and 
       len(box_dims) == 6 and np.issubdtype(box_dims.dtype, np.floating)):
      raise ValueError('Box dimensions [xlo xhi ylo yhi zlo zhi] array "box_dims" '+
                       'must be a floating numpy 1D array of length 6')
      
   if tilt_values is not None:
      if not (isinstance(tilt_values, np.ndarray) and len(np.shape(tilt_values)) == 1 and 
              len(tilt_values) == 3 and np.issubdtype(tilt_values.dtype, np.floating)):
         raise ValueError('Simulation box tilt factors [xy, xz, yz] "tilt_values" must be a '+
                          'floating numpy 1D array of length 3')
      
   if masses is not None:
      if not (isinstance(masses, np.ndarray) and len(np.shape(masses)) == 1 and 
              len(masses) == n_types and np.issubdtype(masses.dtype, np.floating)):
         raise ValueError('Mass array "masses" must be a floating numpy 1D array '+ 
                          'of length equal to the maximum number of atom types '+
                          '"n_types"')
         
   if velocities is not None:
      if not (isinstance(velocities, np.ndarray) and len(np.shape(velocities)) == 2 and 
          len(velocities) == n_atoms and np.shape(velocities)[1] == 3 and 
          np.issubdtype(velocities.dtype, np.floating)):
         raise ValueError('Atom velocities "velocities" must be a floating numpy '+
                          '2D array of length equal to the number of atoms, '+
                          'with 3 columns, each column corresponding to '+
                          'a spatial dimension in 3D space')

   if not isinstance(file_path, str):
      raise ValueError('file_path must be a string')

   f = open(file_path, 'w')
   
   if file_description is None:
      f.write('LAMMPS data file\n\n')
   else:
      if not isinstance(file_description, str):
         raise ValueError('optional argument file_description must be a string')
      elif '\n' in file_description:
         if ''.join(file_description.split('\n')[1:]) == '':
            file_description = file_description.split('\n')[0]
         else:
            raise ValueError('file_description must not contain a next line in '+
                             'middle of the string')
      else:
         f.write(f'{file_description}\n\n')
         
   f.write('{0} atoms\n'.format(n_atoms))
   f.write('{0} atom types\n\n'.format(n_types))
   f.write('{0} {1} xlo xhi\n'.format(box_dims[0], box_dims[1]))
   f.write('{0} {1} ylo yhi\n'.format(box_dims[2], box_dims[3]))
   f.write('{0} {1} zlo zhi\n\n'.format(box_dims[4], box_dims[5]))
   if tilt_values is not None:
      f.write('{0} xy xz yz\n\n'.format(' '.join(map(str, tilt_values))))
      
   if masses is not None:
      f.write('Masses\n\n')
      f.write('\n'.join([f'{i} {masses[i]}' for i in np.arange(1, n_types+1)])+
              '\n\n')
      
   f.write('Atoms\n\n')
   if image_flags is None:
      f.write('\n'.join([' '.join(map(str, ( [ids[i], atom_types[i]] + list(xyz[i]) ) )) 
                         for i in range(len(xyz))]))
   else:
      f.write('\n'.join([' '.join(map(str, ( [ids[i], atom_types[i]] + list(xyz[i]) 
                                             + list(image_flags[i]) ) )) 
                         for i in range(len(xyz))]))
   
   if velocities is not None:
      f.write('Velocities\n\n')
      f.write('\n'.join([' '.join(map(str, ( [ids[i]] + list(velocities[i]) ) )) 
                         for i in range(len(xyz))]))
   
   f.close()

######################################################################################################################################

def read_data_from_LAMMPS_dump_file(file_path:str,
                                    desired_fields:dict or list or 'all'=None,
                                    nonstandard_field_names:dict=None):
   
   standard_field_names = ['id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz',
                           'fx', 'fy', 'fz', 'vx', 'vy', 'vz', 'c_peperatom', 
                           'c_keperatom', 'c_stressperatom[1]', 'c_stressperatom[2]', 
                           'c_stressperatom[3]', 'c_stressperatom[4]', 
                           'c_stressperatom[5]', 'c_stressperatom[6]']
   
   nonstandard_field_names0 = {'id': 'ParticleIdentifier', 'type': 'ParticleType',
                               'x': 'Position.x', 'y': 'Position.y', 'z': 'Position.z',
                               'ix': 'Image.x', 'iy': 'Image.y', 'iz': 'Image.z',
                               'vx': 'Velocity.x', 'vy': 'Velocity.y', 'vz': 'Velocity.z',
                               'fx': 'Force.x', 'fy': 'Force.y', 'fz': 'Force.z',
                               'c_peperatom': 'PotEng', 'c_keperatom': 'KinEng',
                               'c_stressperatom[1]': 'Stress.xx', 'c_stressperatom[2]': 'Stress.yy',
                               'c_stressperatom[3]': 'Stress.zz', 'c_stressperatom[4]': 'Stress.xy',
                               'c_stressperatom[5]': 'Stress.xz', 'c_stressperatom[6]': 'Stress.yz'
                               }
   if nonstandard_field_names is not None:
      if not set(nonstandard_field_names.keys()).issubset(set(standard_field_names)):
         raise ValueError('You can use "nonstandard_field_names" to specify '+
                          'field names that the dump file uses only for these '+
                          f'few standard fields:\n{standard_field_names}')
      nonstandard_field_names0.update(nonstandard_field_names)
   nonstandard_field_names = nonstandard_field_names0
   
   default_datatypes = {'id': ('ParticleIdentifier', 'i8'),
                        'type': ('ParticleType', 'i4'),
                        'c_peperatom': ('PotentialEnergy', 'f8'),
                        'c_keperatom': ('KineticEnergy', 'f8'),
                        'x': ('Position.x', 'f8'),
                        'y': ('Position.y', 'f8'),
                        'z': ('Position.z', 'f8'),
                        'vx': ('Velocity.x', 'f8'),
                        'vy': ('Velocity.y', 'f8'),
                        'vz': ('Velocity.z', 'f8'),
                        'fx': ('Force.x', 'f8'),
                        'fy': ('Force.y', 'f8'),
                        'fz': ('Force.z', 'f8'),
                        'ix': ('Image.x', 'i4'),
                        'iy': ('Image.y', 'i4'),
                        'iz': ('Image.z', 'i4'),
                        'c_stressperatom[1]': ('Stress.xx', 'f8'),
                        'c_stressperatom[2]': ('Stress.yy', 'f8'),
                        'c_stressperatom[3]': ('Stress.zz', 'f8'),
                        'c_stressperatom[4]': ('Stress.xy', 'f8'),
                        'c_stressperatom[5]': ('Stress.xz', 'f8'),
                        'c_stressperatom[6]': ('Stress.yz', 'f8')
                       }     
   
   import copy
   
   if file_path.strip()[-3:] == '.gz':
      import gzip
      f = gzip.open(file_path, 'rt')
   else:
      f = open(file_path, 'rt')
      
   for i, line in enumerate(f):
      
      if i==1:
         step = int(line)
      elif i==3:
         n_atoms = int(line)
      elif i==4:
         LAMMPS_boundary_cond = line.split()[-3:]
      elif i==5:
         xlims = list(map(float, line.split()))
      elif i==6:
         ylims = list(map(float, line.split()))
      elif i==7:
         zlims = list(map(float, line.split()))
      elif i==8:
         fields = line.split()[2:]
      elif i==9:
         break
      
   f.close()
   
   
   if np.allclose(list(map(len, [xlims, ylims, zlims])), 2):
      tilt_factors = None
   elif np.allclose(list(map(len, [xlims, ylims, zlims])), 3):
      tilt_factors = {'xy': xlims[-1], 'xz': ylims[-1], 'yz': zlims[-1]}
      xlims = xlims[:2]; ylims = ylims[:2]; zlims = zlims[:2]
   else:
      raise RuntimeError('All the three list xlims, ylims and zlims must have '+
                         'the same length, either 2 or 3')
      
   
   if desired_fields is None:
      return step, n_atoms, LAMMPS_boundary_cond, xlims, ylims, zlims, tilt_factors
   elif desired_fields == 'all':
      desired_fields = copy.copy(fields)
   elif isinstance(desired_fields, list):
      tmp = []
      for df in desired_fields:
         if df in tmp:
            pass
         else:
            tmp.extend([df])
      desired_fields = tmp
   elif isinstance(desired_fields, dict):
      pass
   else:
      raise ValueError('"desired_fields" must either be a dictionary or a list '+
                       'or "all" or None if only system info is needed')
      
       
   composite_fields = {'Position': ['x', 'y', 'z'], 'Image': ['ix', 'iy', 'iz'], 
                       'Velocity': ['vx', 'vy', 'vz'], 
                       'Force': ['fx', 'fy', 'fz'], 
                       'Stress': ['c_stressperatom[1]', 'c_stressperatom[2]',
                                  'c_stressperatom[3]', 'c_stressperatom[4]',
                                  'c_stressperatom[5]', 'c_stressperatom[6]']}
   
   for df in desired_fields:
      if isinstance(desired_fields, list):
         if df in composite_fields:
            desired_fields[desired_fields.index(df)] = composite_fields[df]
      elif isinstance(desired_fields, dict):
         if df in composite_fields:
            desired_fields.update(dict(zip(composite_fields[df], 
                                           [None]*len(composite_fields[df]))))
            del desired_fields[df]
            
   if isinstance(desired_fields, list):
      desired_fields = sum([[df] if isinstance(df, str) else df 
                            for df in desired_fields], [])
      
   for x in nonstandard_field_names:
      if nonstandard_field_names[x] in fields:
         fields[fields.index(nonstandard_field_names[x])] = x
      if nonstandard_field_names[x] in desired_fields:
         if isinstance(desired_fields, list):
            desired_fields[desired_fields.index(
                                    nonstandard_field_names[x])] = x
         elif isinstance(desired_fields, dict):
            if desired_fields[nonstandard_field_names[x]] is None:
               desired_fields.update({x: default_datatypes[x]})
            else:
               desired_fields.update({x: desired_fields[nonstandard_field_names[x]]})
            del desired_fields[nonstandard_field_names[x]]
            
   if isinstance(desired_fields, list):
      if set(desired_fields).issubset(set(standard_field_names)):
         desired_fields = dict([(x, default_datatypes[x]) for x in desired_fields])
      else:
         raise ValueError('If "desired_fields" is a list then it must refer to '+
                          f'these few standard fields:\n{standard_field_names}.\n'+
                          'This is because the datatypes of these standard '+
                          'fields are already hard-coded. Currently "desired_fields" '+
                          f'refer to the following:\n{desired_fields}'
                          )  
         
   col_indices = []; field_dtype = []
   
   for df in desired_fields:
      if df in fields:
         col_indices = col_indices + [fields.index(df)]
         field_dtype = field_dtype + [desired_fields[df]]
      else:
         raise ValueError(f'Desired field {df} is not tabulated in the input '+
                          'dump file.\n The fields in the input dump file are \n'+
                          f'{fields}')
         
   field_dtype = np.dtype(field_dtype)
         
   dump_data = np.loadtxt(file_path, dtype = field_dtype,
                          skiprows=9, usecols=col_indices)
   
   return ( step, n_atoms, LAMMPS_boundary_cond, xlims, ylims, zlims, tilt_factors, 
           dump_data ) 


######################################################################################################################################

def write_lammps_dump_file(file_path, step, n_atoms, LAMMPS_boundary_cond, 
                           xlims, ylims, zlims, dump_data, tilt_factors=None):
   
   f_str = f'ITEM: TIMESTEP\n{step}\n'
   f_str += f'ITEM: NUMBER OF ATOMS\n{n_atoms}\n'
   if tilt_factors is None:
      f_str += 'ITEM: BOX BOUNDS '+' '.join(LAMMPS_boundary_cond)+'\n'
      f_str += ' '.join(map(str, xlims)) + '\n'
      f_str += ' '.join(map(str, ylims)) + '\n'
      f_str += ' '.join(map(str, zlims)) + '\n'
   elif ( isinstance(tilt_factors, dict) and 
          ( set(tilt_factors.keys()) == {'xy', 'xz', 'yz'} ) and
          all([isinstance(tilt_factors[x], float) for x in ['xy', 'xz', 'yz']]) ):
      f_str += 'ITEM: BOX BOUNDS xy xz yz '+' '.join(LAMMPS_boundary_cond)+'\n'
      f_str += ' '.join(map(str, xlims+tilt_factors['xy'])) + '\n'
      f_str += ' '.join(map(str, ylims+tilt_factors['xz'])) + '\n'
      f_str += ' '.join(map(str, zlims++tilt_factors['yz'])) + '\n'
   else:
      raise ValueError("Tilt factors must be a dictionary with keys "+
                       "{xy, xz, yz} and floating values")
      
   name_correspondence = dict([
      ('ParticleIdentifier', 'id'), ('ParticleType', 'type'),
      ('Position.x', 'x'), ('Position.y', 'y'), ('Position.z', 'z'),
      ('Velocity.x', 'vx'), ('Velocity.y', 'vy'), ('Velocity.z', 'vz'),
      ('Force.x', 'fx'), ('Force.y', 'fy'), ('Force.z', 'fz'),
      ('Image.x', 'ix'), ('Image.y', 'iy'), ('Image.z', 'iz')  ])
      
   if dump_data.dtype.names is None:
      raise ValueError('"dump_data" must be a structured numpy array')
   else:
      field_names = []
      for name in dump_data.dtype.names:
         if name in name_correspondence:
            field_names.extend([name_correspondence[name]])
         else:
            field_names.extend([name])
      tmp = []
      for name in field_names:
         if name in tmp:
            pass
         else:
            tmp.extend([name])
      field_names = tmp
         
      f_str += 'ITEM: ATOMS '+' '.join(field_names)
      
   np.savetxt(file_path, dump_data, fmt='%.15g', header=f_str, comments='')
      
#######################################################################################################################################

def read_vtk_file(filename, elements):

   tree = ET.parse(filename)
   root = tree.getroot()
   attribs = dict([(key, root.find(elements[key]).attrib) for key in elements.keys()]) 
   texts = dict([(key, root.find(elements[key]).text) for key in elements.keys()]) 
   
   return attribs, texts

################################################################################################################################

def write_vtk_file(points, connectivity, point_data, cell_data, file_path, file_name):

   """
   VTK file dumper for UnstructuredGrid 3D data in XML format (thus file_name must have .vtu extension)
   
   All cell type assumed to be tetrahedral (but it can be changed in the code if needed)
   points = numpy array (Npoints, 3): x, y, z coordinates of points
   connectivity = numpy array (Ncells, 4): indices of the four cell vertices
   point_data = dict {name_str1: data1 numpy array of shape (Npoints, dim1), name_str2: data2 numpy array of shape (Npoints, dim2)....} where dim is the dim of the data
   cell_data = dict {name_str1: data1 numpy array of shape (Ncells, dim1), name_str2: data2 numpy array of shape (Ncells, dim2)....} where dim is the dim of the data
   file_path = string: absolute or relative path which will be created, in case it does not exist (Note: the file_path string MUST end will a '/')
   file_name = string: the output .vtu file which will be written in the file_path

   """

   if file_path[-1] != '/':
      raise RuntimeError('The "file_path" string MUST end will a "/"')

   if not os.path.isdir(file_path):
      os.makedirs(file_path)

   if file_name[-4:] != '.vtu':
      raise RuntimeError('This code writes VTK file for UnstructuredGrid data. So the filename extension must be ".vtu"')

   argument_type = [(points, np.ndarray), (connectivity, np.ndarray), (point_data, dict), (cell_data, dict), (file_path, str), (file_name, str)]

   if not np.array([isinstance(i[0], i[1]) for i in argument_type]).all():
      raise RuntimeError('TypeError: Input argument to the funtion is of the wrong type')
   
   
   #if np.result_type(connectivity) != np.uint32:
      #raise RuntimeError('The dtype of the numpy array "connectivity" must be "int"')

   f = open(file_path + file_name, 'w')
   double_space = '  '
   f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
   f.write(double_space + '<UnstructuredGrid>\n')
   f.write((2*double_space) + '<Piece NumberOfPoints="{0}" NumberOfCells="{1}">\n'.format(np.shape(points)[0], np.shape(connectivity)[0]))
   f.write((3*double_space) + '<Points>\n')
   f.write((4*double_space) + '<DataArray type="Float64" NumberOfComponents="3" Name="positions" format="ascii">\n')
   f.write('\n'.join([((5*double_space) + ' '.join(map(str, row))) for row in points]) + '\n')
   f.write((4*double_space) + '</DataArray>\n')
   f.write((3*double_space) + '</Points>\n')
   f.write((3*double_space) + '<Cells>\n')
   f.write((4*double_space) + '<DataArray type="Int32" Name="connectivity" format="ascii">\n')
   f.write((5*double_space) + ' '.join(map(str, connectivity.flatten())) + '\n')
   f.write((4*double_space) + '</DataArray>\n')
   f.write((4*double_space) + '<DataArray type="Int32" Name="offsets" format="ascii">\n')
   f.write((5*double_space) + ' '.join(map(str, 4*(np.array(range(np.shape(connectivity)[0]))+1))) + '\n')
   f.write((4*double_space) + '</DataArray>\n')
   f.write((4*double_space) + '<DataArray type="UInt32" Name="types" format="ascii">\n')
   f.write((5*double_space) + ' '.join(map(str, 10*np.ones(np.shape(connectivity)[0], dtype = np.int))) + '\n')
   f.write((4*double_space) + '</DataArray>\n')
   f.write((3*double_space) + '</Cells>\n')
   f.write((3*double_space) + '<PointData Scalars="scalars">\n')
   if len(point_data) != 0:
      for key in point_data:
         if not isinstance(point_data[key], np.ndarray):
            raise RuntimeError('Values in dict "point_data" must be numpy arrays')
         elif np.result_type(point_data[key]) == 'object':
            raise RuntimeError('Values in dict "point_data" must not be numpy arrays of dtype "object"')
         elif point_data[key].size == 0:
            raise RuntimeError('Values in dict "point_data" must not be empty numpy arrays')
         elif len(np.shape(point_data[key])) > 2:
            raise RuntimeError('Values in dict "point_data" must be either 1D or 2D arrays')
         elif len(np.shape(point_data[key])) == 1:
            f.write((4*double_space) + '<DataArray type="{0}" Name="{1}" Format="ascii">\n'.format(np.result_type(point_data[key]).name.capitalize(), key))
            f.write((5*double_space) + ' '.join(map(str, point_data[key])) + '\n')
            f.write((4*double_space) + '</DataArray>\n')
         elif len(np.shape(point_data[key])) == 2:
            if np.shape(point_data[key])[1] == 1:
               raise RuntimeError('Values in dict "point_data" must not be column numpy 2D arrays, meaning of shape (., 1)')
            else:
               f.write((4*double_space) + '<DataArray type="{0}" Name="{1}" NumberOfComponents="{2}" Format="ascii">\n'.format(np.result_type(point_data[key]).name.capitalize(), key, np.shape(point_data[key])[1]))
               f.write('\n'.join([(5*double_space) + ' '.join(map(str, row)) for row in point_data[key]]) + '\n')
               f.write((4*double_space) + '</DataArray>\n')
   f.write((3*double_space) + '</PointData>\n')
   f.write((3*double_space) + '<CellData Scalars="scalars">\n')
   if len(cell_data) != 0:
      for key in cell_data:
         if not isinstance(cell_data[key], np.ndarray):
            raise RuntimeError('Values in dict "cell_data" must be numpy arrays')
         elif np.result_type(cell_data[key]) == 'object':
            raise RuntimeError('Values in dict "cell_data" must not be numpy arrays of dtype "object"')
         elif cell_data[key].size == 0:
            raise RuntimeError('Values in dict "cell_data" must not be empty numpy arrays')
         elif len(np.shape(cell_data[key])) > 2:
            raise RuntimeError('Values in dict "cell_data" must be either 1D or 2D arrays')
         elif len(np.shape(cell_data[key])) == 1:
            f.write((4*double_space) + '<DataArray type="{0}" Name="{1}" Format="ascii">\n'.format(np.result_type(cell_data[key]).name.capitalize(), key))
            f.write((5*double_space) + ' '.join(map(str, cell_data[key])) + '\n')
            f.write((4*double_space) + '</DataArray>\n')
         elif len(np.shape(cell_data[key])) == 2:
            if np.shape(cell_data[key])[1] == 1:
               raise RuntimeError('Values in dict "cell_data" must not be column numpy 2D arrays, meaning of shape (., 1)')
            else:
               f.write((4*double_space) + '<DataArray type="{0}" Name="{1}" NumberOfComponents="{2}" Format="ascii">\n'.format(np.result_type(cell_data[key]).name.capitalize(), key, np.shape(cell_data[key])[1]))
               f.write('\n'.join([(5*double_space) + ' '.join(map(str, row)) for row in cell_data[key]]) + '\n')
               f.write((4*double_space) + '</DataArray>\n')
   f.write((3*double_space) + '</CellData>\n')
   f.write((2*double_space) + '</Piece>\n')
   f.write(double_space + '</UnstructuredGrid>\n')
   f.write('</VTKFile>')
   f.close()

######################################################################################################################################


def save_obj(obj, name):

   with open('{0}.pickle'.format(name), 'wb') as handle:
      pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(name):

   with open('{0}.pickle'.format(name), 'rb') as handle:
      return pickle.load(handle)

###################################################################################################################################




