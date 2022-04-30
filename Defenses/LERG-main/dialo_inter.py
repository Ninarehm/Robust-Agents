from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
import torch
import requests
import json
import random 
import argparse
from perturbation_models import RandomPM
from RG_explainers import LERG_SHAP_log as LERG_S
import matplotlib.pyplot as plt
import numpy as np



def plot_interactions(phi_map,x,y):
	values = np.around([[phi_map[(i,j)].item() for i in range(len(x))] for j in range(len(y))], decimals=2)
	fig = plt.figure()#figsize=(w,h))
	ax = plt.axes()
	im = ax.imshow(values, cmap=plt.get_cmap('Reds'))

	ax.set_xticks(np.arange(len(x)))
	ax.set_yticks(np.arange(len(y)))
	x = [w[:-4] for w in x]
	y = [w[:-4] for w in y]
	ax.set_xticklabels(x, fontsize=11)
	ax.set_yticklabels(y, fontsize=11)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")
	for i in range(len(y)):
	    for j in range(len(x)):
	        text = ax.text(j, i, values[i, j],
	                       ha="center", va="center", color="w")
	plt.savefig('foo.png') 
	plt.close()



if __name__ == '__main__':

	
	device = torch.device("cuda")

	output_toxicity = 0
	generated_adversary = 1
	after_attack_w_adversary = 0
	after_attack_wo_adversary = 0
	before_attack_w_adversary = 0
	before_attack_wo_adversary = 0
	attack_toxicity = 0
	defender_toxicity = 0

	config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")
	tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
	model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

	model.eval()
	model.to(device)


	perturb_f = RandomPM(denoising=False).perturb_inputs
	local_exp  = LERG_S(model, "Hi.", "Hello", perturb_f, tokenizer)
	phi_set, phi_map, input_segments, output_segments = local_exp.get_local_exp()
	print(phi_set)
	print(phi_map)
	print(input_segments)
	print(output_segments)
	plot_interactions(phi_map,input_segments,output_segments)



	