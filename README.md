# Twin grain identification in nanocrystalline specimens

The objective is to identify fcc atoms with twin misorientation with respect to a reference orientation

**grain_segmentation_analysis:**

This module uses the Grain segmentation modifier implemented in OVITO open-source visualization software to partition atoms in a nanocrystal into groups with similar lattice orientation. The Grain ID is appended to the particles properties and the grain orientation is noted in a custom GrainSegmentationAnalysis output file.

**calculate_disorientation.py**

This module compares the grains in a certain snapshot of nanocrystalline deformation with the grains in corresponding position in an initial reference configuration and calculates the misorientation angle. 

This module has three classes: ReadOrientationFile, FindGrainCorrespondence, CalculateDisorientationAngles

class *ReadOrientationFile*: Read the GrainSegmentationAnalysis file generated with the grain_segmentation_analysis module mentioned above. 

class *FindGrainCorrespondence*: Enlists the correspondence by GrainID of grains in the current nanocrystalline configuration with grains in the same location in the initial reference system configuration.

