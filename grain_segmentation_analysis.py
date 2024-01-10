from ovito.io import *
from ovito.pipeline import StaticSource, Pipeline
from ovito.data import *
from ovito.modifiers import *
import numpy as np
import glob
import sys, os, re



def analyse(structure_file_path, output_folder, rmsd_cutoff, no_image_flags,
            target_cell = None):

   # Load the simulation dataset to be analyzed.
   pipeline = import_file(structure_file_path)
   
   data = pipeline.compute()
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
   
   data = pipeline.compute()
   
   box_dims = np.array(list(zip(original_cell[:,-1], 
                                (np.diag(original_cell[:,:3])+
                                 original_cell[:,-1])))).ravel()
   
   f = open((output_folder+'grain_segmentation_analysis_outputs/'+
             'GrainSegmentationAnalysis.'+
             '.'.join(structure_file_path.split('/')[-1].split('.')[:2])+'.txt'), 
            'wt')
   f.write(f'RMSD for PTM analysis = {rmsd_cutoff}\n')
   f.write(f'Number of grains = {data.attributes["GrainSegmentation.grain_count"]}\n')
   f.write(f'Box dimensions = {" ".join(map(str, box_dims))} \n\n')
   f.write('GrainID Grain_orientation \n')
   f.write('\n'.join([str(i+1)+' '+' '.join(map(str, x)) 
                      for i,x in enumerate(data.tables['grains']['Orientation'])])+'\n\n')
   
   if no_image_flags:
      f.write('AtomID AtomType StructureType Grain x y z \n')
      f.write('\n'.join([
         ' '.join(map(str, [data.particles['Particle Identifier'][i]]+
                           [data.particles['Particle Type'][i]]+
                           [data.particles['Structure Type'][i]]+
                           [data.particles['Grain'][i]]+
                           list(data.particles['Position'][i]))) 
                  for i in range(data.particles.count) ]) )
   else:
      f.write('AtomID AtomType StructureType Grain x y z ix iy iz \n')
      f.write('\n'.join([
         ' '.join(map(str, [data.particles['Particle Identifier'][i]]+
                           [data.particles['Particle Type'][i]]+
                           [data.particles['Structure Type'][i]]+
                           [data.particles['Grain'][i]]+
                           list(data.particles['Position'][i])+
                           list(data.particles['Periodic Image'][i]))) 
                  for i in range(data.particles.count) ]) )
   
   f.close()   
   
   
def get_box_dims(structure_file_path):
   
   if structure_file_path[-3:] == '.gz':
      import gzip
      f = gzip.open(structure_file_path, 'rt')
   else:
      f = open(structure_file_path, 'rt')
   
   for i, line in enumerate(f):
      if i < 5:
         continue
      if i == 5:
         xdims = list(map(float, line.split()))
      elif i == 6:
         ydims = list(map(float, line.split()))
      elif i == 7:
         zdims = list(map(float, line.split()))
      else:
         break
      
   return np.c_[np.diag(np.diff(np.c_[xdims, ydims, zdims], axis=0)[0]),
                np.c_[xdims, ydims, zdims][0]] #ovito 3*4 cell matrix 
   
def main():
   
   import argparse
   
   parser = argparse.ArgumentParser()
   parser.add_argument('input_file_path_pattern', type=str)
   parser.add_argument('output_folder', type=str)
   parser.add_argument('--rmsd_cutoff', type=float, required=False, 
                       default=0.15)
   parser.add_argument('--file_num_index', type=int, required=False)
   parser.add_argument('--no_image_flags', action='store_true', required=False)
   
   args = parser.parse_args()
   
   input_file_path_pattern = args.input_file_path_pattern
   output_folder = args.output_folder
   rmsd_cutoff = args.rmsd_cutoff
   file_num_index = args.file_num_index
   
   if output_folder[-1] != '/':
      output_folder = output_folder + '/'
   
   if not os.path.isdir(output_folder):
      os.mkdir(output_folder)
      
   if not os.path.isdir(output_folder+'grain_segmentation_analysis_outputs/'):
      os.mkdir(output_folder+'grain_segmentation_analysis_outputs/')
      
   files = glob.glob(input_file_path_pattern)
   
   file_nums = []
   for f in files:
      n = re.findall('\d+', f.split('/')[-1])
      if len(n) == 0:
         raise RuntimeError('Cannot screen pattern')
      elif len(n) == 1:
         file_nums.extend([int(n[0])])
      else:
         file_nums.extend([int(n[file_num_index])])
         
   files = np.array(files)[np.argsort(file_nums)] 
   
   file_ind = int(os.environ['SLURM_ARRAY_TASK_ID'])
   
   if file_ind == 0:
      analyse(files[file_ind], output_folder, rmsd_cutoff, args.no_image_flags)
   else:
      initial_cell = get_box_dims(files[0])
      analyse(files[file_ind], output_folder, rmsd_cutoff, args.no_image_flags,
              target_cell=initial_cell)

if __name__ == '__main__':
   main()

