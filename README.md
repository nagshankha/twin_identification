# Twin grain identification in nanocrystalline specimens

The objective is to identify fcc atoms with twin misorientation with respect to a reference orientation

**grain_segmentation_analysis:**

This module uses the Grain segmentation modifier implemented in OVITO open-source visualization software to partition atoms in a nanocrystal into groups with similar lattice orientation. The Grain ID is appended to the particles properties and the grain orientation is noted in a custom GrainSegmentationAnalysis output file.

**calculate_disorientation.py:**

This module compares the grains in a certain snapshot of nanocrystalline deformation with the grains in corresponding position in an initial reference configuration and calculates the misorientation angle. 

This module has three classes: ReadOrientationFile, FindGrainCorrespondence, CalculateDisorientationAngles

- class *ReadOrientationFile*: Read the GrainSegmentationAnalysis file generated with the grain_segmentation_analysis module mentioned above. 

- class *FindGrainCorrespondence*: Enlists the correspondence by GrainID of grains in the current nanocrystalline snapshot with grains in the same location in the initial reference system snapshot.

- class *CalculateDisorientationAngles*: Using the orientation quarternion for each grain stores in the GrainSegmentationAnalysis file and the grain correspondences calculated in the class FindGrainCorrespondence, the misorientation angle is calculated for every grain in the current nanocrystalline snapshot with respect to the corresponding grains in the initial reference snapshot. It also creates a dump file with atom ID, types, positions, image flags, Grain ID and the calculated misorientation angle.

**analyse_faults.py:**

This module uses the dump files generated with misorientation angles by calculate_disorientation.py to identify fcc atoms with twin orientation.

**io_files.py**

A module for reading and writing LAMMPS data and dump files and VTK files. 
