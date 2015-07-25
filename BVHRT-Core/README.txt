------------------------------------------------------------------------------------------------------------------------

                         ATRBVH - Agglomerative Treelet Restructuring Bounding Volume Hierarchy                         
                                        Copyright 2015 Leonardo Rodrigo Domingues                                       

------------------------------------------------------------------------------------------------------------------------

In this folder, you will find the full implementation for ATRBVH, as described in:

	"Bounding Volume Hierarchy Optimization through Agglomerative Treelet Restructuring",
	Leonardo R. Domingues and Helio Pedrini,
	High-Performance Graphics 2015

You will also find an implementation of TRBVH, which was created from the information provided in the original paper:

	"Fast Parallel Construction of High-Quality Bounding Volume Hierarchies",
	Tero Karras and Timo Aila,
	High-Performance Graphics 2013,
	https://research.nvidia.com/publication/fast-parallel-construction-high-quality-bounding-volume-hierarchies

An implementation of LBVH can also be found here, as described in:

	"Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees",
	Tero Karras,
	High-Performance Graphics 2012,
	https://research.nvidia.com/publication/maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees
	

If you are going to use this project, please note that both TRBVH and the idea of using an agglomerative treelet 
restructuring are patented:

	"Bounding volume hierarchies through treelet restructuring",
	US 20140365532 A1,
	http://www.google.com/patents/US20140365532

	"Agglomerative treelet restructuring for bounding volume hierarchies",
	US 20140365529 A1,
	http://www.google.com/patents/US20140365529

The source code contained in this folder is licensed under the MIT License, please check 'LICENSE.txt' for the full 
license.

------------------------------------------------------------------------------------------------------------------------

INSTRUCTIONS:


Either run '../gpu-ray-traversal/rt.exe' directly or open '../BVHRT.sln' and compile the project. The executable will 
be generated in the 'gpu-ray-traversal' folder.

To change the algorithms parameters, edit '../gpu-ray-traversal/bvhrt.cfg'. Possible commands are:
	-'lbvh64' or 'lbvh', to create the initial LBVH tree using 30-bit or 63-bit morton codes, respectively.
	-[optional] 'atrbvh' or 'trbvh', to optimize the tree using ATRBVH or TRBVH, respectively.
	-[optional] 'treeletSize=n', to configure the treelet size used in ATRBVH and TRBVH. Default is 9.
	-[optional] 'iterations=n', to configure the number of iterations used in ATRBVH and TRBVH. Default is 3.
	-[optional] 'collapse', to identify that the final tree should be collapsed, allowing more than one triangle per 
	leaf in order to reduce its SAH.

Commands must be separated by a white space. Example configuration files (must not use quotes inside the actual config 
file):

'lbvh collapse' - LBVH
'lbvh64 atrbvh treeletSize=9 iterations=2 collapse' - ATRBVH as described in the paper
'lbvh64 trbvh treeletSize=7 iterations=3 collapse' - TRBVH as described in the paper

For more information on the ray tracer used, check '../gpu-ray-traversal/README'.

------------------------------------------------------------------------------------------------------------------------