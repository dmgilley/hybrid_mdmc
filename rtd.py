#!/usr/bin/env python3
# Author
#    Vignesh Sathyaseelan
#    vsathyas@purdue.edu

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import os, argparse, itertools, sys
import matplotlib.pyplot as plt

#MD traj plot
import glob 
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D 

# Argparse initializations
parser = argparse.ArgumentParser(description = 'Molecular Dynamics Residence Time')
parser.add_argument('data_file', type = str, help = 'Traj file')
parser.add_argument('-N',dest='num_mol', type=int, default = 100, help = 'Number of molecules in box')
parser.add_argument('-n',dest='num_atoms', type=int, default = 3, help = 'Number of atoms in molecule')
parser.add_argument('-n_d',dest='num_disc', type=int, default = 3, help = 'Number of voxels on each axis')
args = parser.parse_args()

def main(argv):

	array_of_tuples=[(x,args.num_mol,args.num_atoms,args.num_disc) for x in args.data_file.split()]
	
	#par_fun(args.data_file,args.num_mol,args.num_atoms,args.num_disc) 
	with Pool(mp.cpu_count()) as p:
		result = p.starmap(par_fun,array_of_tuples)

	return args.data_file.split(), result  ##(result will have a set (ordered voxel_list, diffusion score))

def par_fun(traj_file,num_mol,num_atoms,num_disc):

	traj_all,box_dims = read_traj(traj_file,num_mol,num_atoms)
	voxels = gen_voxel(box_dims,num_disc)
	voxel_map_all = map_voxel(traj_all,voxels,box_dims,num_atoms)
	voxel,dif_score = get_diff_time(voxel_map_all)

	fn = traj_file.split('/')[-1].split('.')[0]
	traj_anim(traj_all,args.num_atoms,box_dims,outname=f'traj_{fn}.gif')
	plot_bar(dif_score,voxel,fig_name=f'dif_t_{fn}.png',xlabel='diffusion time (unit timestep)',ylabel='voxel')

	return (voxel,dif_score)

def divide_chunks(l, n):
     
    for i in range(0, len(l), n):
        yield l[i:i + n]

def read_traj(data_file,num_mol,num_atoms):
	
	traj_all = []
	with open(data_file, 'r') as datafile:
		data_all = datafile.readlines()
		data_all = list(divide_chunks(data_all,(num_mol*num_atoms)+9))  #num atoms + 9
		for data in data_all:
			timestep = data[1]
			box_dims = np.array([[float(_) for _ in x.rstrip().split()] for x in data[5:8]])
			traj_all += [data[9:]]
	
	return traj_all,box_dims

def gen_voxel(box_dims,num_disc):

	x = np.linspace(box_dims[0,0],box_dims[0,1], num_disc+1)
	y = np.linspace(box_dims[1,0],box_dims[1,1], num_disc+1)
	z = np.linspace(box_dims[2,0],box_dims[2,1], num_disc+1)

	xlims = [(x[i],x[i+1]) for i in range(len(x)-1)]
	ylims = [(y[i],y[i+1]) for i in range(len(y)-1)]
	zlims = [(z[i],z[i+1]) for i in range(len(z)-1)]

	voxels = np.array([p for p in itertools.product(xlims,ylims,zlims)])

	return voxels

def map_voxel(traj_all, voxels, box_dims,num_atoms): 
	
	voxel_map_all = []
	for count_traj,traj in enumerate(traj_all):

		#create com
		com = [] 
		for i in list(divide_chunks(traj,num_atoms)):   
			molecule = np.array([[float(_) for count_,_ in enumerate(x.rstrip().split()) if count_ in [2,3,4]] for x in i])
			com += [np.mean(molecule,axis=0)]

		#account for outside voxel
		for count_i,i in enumerate(com):
			for count_j,j in enumerate(i):
				if j>box_dims[0][1]: com[count_i][count_j] = box_dims[0][1]
				elif j < box_dims[0][0]: com[count_i][count_j] = box_dims[0][0]

		#voxel map
		voxel_map = []
		for j,_ in enumerate(com):
			for count_i,i in enumerate(voxels):
				if com[j][0] >= i[0][0] and com[j][0] <= i[0][1]: 
					if com[j][1] >= i[1][0] and com[j][1] <= i[1][1]:
						if com[j][2] >= i[2][0] and com[j][2] <= i[2][1]:
							voxel_map += [count_i]

		voxel_map_all += [voxel_map]

	return voxel_map_all

def get_diff_time(voxel_map_all):

	dif_dict = {}
	for mol_count in range(len(voxel_map_all[0])):

		#print(np.array(voxel_map_all)[:,mol_count])

		mol_loc = np.array(voxel_map_all)[:,mol_count]
		u_e = list(set(mol_loc))

		for e in u_e:

			len_e,counter,len_list = 0, 0, []
			
			for count_i,i in enumerate(mol_loc):
				
				if i == e:
					counter = 1
					len_e += 1
				
				elif  i != e and counter == 1:
					len_list += [len_e]
					len_e,counter = 0,0
			
			if counter == 1 :
				len_list += [len_e]

			if e not in dif_dict:dif_dict[e] = [np.mean(np.array(len_list))]
			else:dif_dict[e] += [np.mean(np.array(len_list))]

			#print(e)
			#print(len_list)	
		#quit()

	voxel = []
	dif_t = []
	for i in dif_dict:
		voxel += [i]
		dif_t += [np.mean(dif_dict[i])]

	dif_t = [dif_t[i] for i in np.argsort(voxel)]
	voxel.sort()

	return voxel,dif_t

def plot_bar(x,lable,fig_name,xlabel,ylabel):
    
    fig = plt.figure(dpi=500, figsize=(3,2)) 
    ax = plt.subplot(111)
    
    index = [count_i for count_i,i in enumerate(lable)]
    ax.barh(index, x, align='center',color='#00AF00',alpha=0.5,linewidth=20)
    plt.scatter(np.array(x)+0.005*np.ones(len(x)),index,marker='o',color='#00AF00',alpha=1.0,s=5)
    plt.yticks(index,lable,fontsize=4)
    plt.xticks(np.arange(0,int(max(x))+2,1),fontsize=4)
    
    [j.set_linewidth(1.5) for j in ax.spines.values()]
    
    plt.xlabel(xlabel,fontsize=5)
    plt.ylabel(ylabel,fontsize=5)
    plt.savefig(fig_name, bbox_inches='tight')

    return

def traj_anim(traj_all,num_atoms,box_dims,outname='traj.gif'):

	for count_traj,traj in enumerate(traj_all):
		com = [] 
		for i in list(divide_chunks(traj,num_atoms)):   
			molecule = np.array([[float(_) for count_,_ in enumerate(x.rstrip().split()) if count_ in [2,3,4]] for x in i])
			com += [np.mean(molecule,axis=0)]
	
		com = np.array(com)

		try:
			os.mkdir('figs')
		except:
			pass
	
		# Creating figure
		fig = plt.figure(figsize = (10, 7))
		ax = plt.axes(projection='3d')
		ax.scatter(com[:,0], com[:,1], com[:,2], color = "green")

		ax.set_xlim(box_dims[0][0],box_dims[0][1])
		ax.set_ylim(box_dims[1][0],box_dims[1][1])
		ax.set_zlim(box_dims[2][0],box_dims[2][1])

		plt.title(f"traj {count_traj}")
		plt.savefig(f'figs/{count_traj}.png')

	for root, dirs, files in os.walk('figs', topdown=False): 
        	pass

	pngs = [int(i.split('.')[0]) for i in files]
	pngs.sort()
	imgs = [f'figs/{x}.png' for x in pngs]

	frames = []
	for i in imgs:
		new_frame = Image.open(i)
		frames.append(new_frame)

	# Save into a GIF file that loops forever
	frames[0].save(outname, format='GIF',
				append_images=frames[0:],
				save_all=True,
				duration=300, loop=0)
	
	'''
	num_disc = 4
	fig = plt.figure()
	ma = np.random.choice([0,1], size=(num_disc-1,num_disc-1,num_disc-1),p=[0.75,0.25])
	ax = fig.gca(projection='3d')
	ax.voxels(ma, edgecolor="k")
	plt.savefig('voxel.png')
	'''

	return

if __name__ == '__main__':
    main(sys.argv[1:])
