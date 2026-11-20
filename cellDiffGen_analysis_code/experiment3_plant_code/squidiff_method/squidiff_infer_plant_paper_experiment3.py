import torch
print("CUDA Available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
import scanpy as sc
print(sc.__version__)
import sys
import os
import anndata
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import anndata
import seaborn as sns
import torch
import torch.distributed as dist
import matplotlib.colors as mcolors

import argparse
import os

import numpy as np
# import torch as th
import torch.distributed as dist
from scipy.stats import gaussian_kde
from Squidiff import dist_util, logger
from Squidiff.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=2400,
        batch_size=16,
        use_ddim=False,
        model_path="plant_exp3_squidiff_train_checkpoints/model.pt",
    )
    defaults.update(model_and_diffusion_defaults())

    return defaults

def stochastic_encode(
    model,diffusion, x, t, model_kwargs):
    """
    ddim reverse sample
    """
    sample = x
    sample_t = []
    xstart_t = []
    T = []
    indices = list(range(t))

    for i in indices:
        timestep = torch.full((x.shape[0],), i, device='cuda').long()
        with torch.no_grad():
            out = diffusion.ddim_reverse_sample(model, 
                                                sample, 
                                                timestep, 
                                                model_kwargs=model_kwargs)
            sample = out['sample']
            sample_t.append(sample)
            xstart_t.append(out['pred_xstart'])
            T.append(timestep)

    return {
    'sample': sample,
    'sample_t': sample_t,
    'xstart_t': xstart_t,
    'T': T,
}

# def display_reconst(df_true,
#                     df_pred,
#                     density=False,
#                     den_all = False,
#                     marker_genes=None,
#                     sample_rate=0.1,
#                     size=(4, 4),
#                     spot_size=1,
#                     title=None,
#                     x_label='',
#                     y_label='',
#                     ):
#     """
#     Scatter plot - raw gexp vs. reconstructed gexp
#     """
#     assert 0 < sample_rate <= 1, \
#         "Invalid downsampling rate for reconstruct scatter plot: {}".format(sample_rate)

#     if marker_genes is not None:
#         marker_genes = set(marker_genes)

#     df_true_sample = df_true.sample(frac=sample_rate, random_state=0)
#     df_pred_sample = df_pred.loc[df_true_sample.index]

#     plt.rcParams["figure.figsize"] = size
#     plt.figure(dpi=800)
#     ax = plt.gca()

#     xx = df_true_sample.T.to_numpy().flatten()
#     yy = df_pred_sample.T.to_numpy().flatten()

#     if density and not den_all:
#         for gene in df_true_sample.columns:
#             try:
#                 gene_true = df_true_sample[gene].values
#                 gene_pred = df_pred_sample[gene].values
#                 gexp_stacked = np.vstack([df_true_sample[gene].values, df_pred_sample[gene].values])

#                 z = gaussian_kde(gexp_stacked)(gexp_stacked)
#                 ax.scatter(gene_true, gene_pred, c=z,vmin=0,cmap='Spectral_r',
#                            s=spot_size, alpha=0.5)
#             except np.linalg.LinAlgError as e:
#                 pass

#     elif marker_genes is not None:
#         color_dict = {True: 'red', False: 'green'}
#         gene_colors = np.vectorize(
#             lambda x: color_dict[x in marker_genes]
#         )(df_true_sample.columns)
#         colors = np.repeat(gene_colors, df_true_sample.shape[0])

#         ax.scatter(xx, yy, c=colors, s=spot_size, alpha=0.5)

#     elif density and den_all:
#         gene_true = df_true_sample.values.flatten()
#         gene_pred = df_pred_sample.values.flatten()
#         gexp_stacked = np.vstack([df_true_sample.values.flatten(), df_pred_sample.values.flatten()])
#         z = gaussian_kde(gexp_stacked)(gexp_stacked)
#         ax.scatter(gene_true, gene_pred, c=z,vmin=0,cmap='Spectral_r',rastered=True,
#                            s=spot_size, alpha=0.5)
#     else:
#         ax.scatter(xx, yy, s=spot_size,rastered=True,alpha=0.5)
    
    
        
#     min_val = min(xx.min(), yy.min())
#     max_val = max(xx.max(), yy.max())
#     #ax.set_xlim(min_val, 400)
    
#     #ax.set_ylim(min_val, 400)
#     ax.set_aspect('equal',adjustable='box')
#     ax.axis('equal')
#     ax.set_xlim(0, max_val)
#     ax.set_ylim(min_val-0.2, max_val)
#     plt.suptitle(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.legend()
#     plt.grid(True)
#     print("almost save")
#     plt.savefig("compare_predict_and_true_about_training_data.pdf", bbox_inches="tight")
#     # plt.show()

def slerp(start, direction, weight):
    """Spherical linear interpolation using a start tensor and a direction tensor"""
    end = start + direction
    
    # Correct usage of np.linalg.norm
    start_norm = start / np.linalg.norm(start, ord=2, axis=-1, keepdims=True)
    end_norm = end / np.linalg.norm(end, ord=2, axis=-1, keepdims=True)
    
    dot_product = np.sum(start_norm * end_norm, axis=-1, keepdims=True)
    omega = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    so = np.sin(omega)
    interpolated = (np.sin((1.0 - weight) * omega) / so) * start + (np.sin(weight * omega) / so) * end
    
    return interpolated

##add noise
def sample_around_point(point, num_samples=1000, scale=0.2):
    return point + scale * np.random.randn(num_samples, point.shape[0])

def get_interp(label_end, label_start, model, diffusion, data, args, num = 10):
    z_sem = model.encoder(torch.tensor(data.X.toarray()).to('cuda'),
                        #label = torch.tensor(data.obs.values, dtype = torch.float32).to('cuda')
                        )
    label = torch.tensor(data.obs["Group"].values, dtype=torch.long, device="cuda")
    print("label!!!",label)
    logger.log("sampling...")
    direct_z_sem = z_sem[label==label_end].mean(axis=0) - z_sem[label==label_start].mean(axis=0)

    z_sem_start = model.encoder(torch.tensor(data[data.obs['Group']==0].X.toarray()).to('cuda'))
    # x_T_start = stochastic_encode(model, diffusion,
    #                         x = torch.tensor(data[data.obs['Group']==0].X.toarray(), dtype = torch.float32).to('cuda'),
    #                         t = 999,
    #                         model_kwargs={'x_start':torch.tensor(data[data.obs['Group']==0].X.toarray()).to('cuda'),
    #                                       'z_mod':z_sem_start}
    #                        )['sample']
 
    interp_days = np.linspace(12,21,num)

    interp_weights = np.linspace(0,1,num)
    weights_all = []
    interp_all = []
    #-1day = 0day data
    for (i,day_) in zip(list(interp_weights), list(interp_days)):
        print("weight,interp_days:",i,day_)

        z_sem_interp_day_ = z_sem_start.detach().cpu().numpy().mean(axis=0) + direct_z_sem.detach().cpu().numpy() * i
        z_sem_interp_day_ = sample_around_point(z_sem_interp_day_)
        # xT_interp_day_ = slerp(x_T_start.detach().cpu().numpy().mean(axis=0), direct_xT.detach().cpu().numpy(),  i)
        # xT_interp_day_ = sample_around_point(xT_interp_day_)
        
        logger.log("sampling...")
        
        sample_fn = (diffusion.p_sample_loop if not args['use_ddim'] else diffusion.ddim_sample_loop)
        # if day_ == -1:
        #     #noise = x_T_start
        #     noise = None
        # else:
        noise = None
        #noise = None

        # PREDICT FUNCTION!!!when predict only use day 0 here mark as -1 data
        #"-1 group” = baseline/anchor
        #“1 group” = interpolated/extrapolated
        sample_interp_ = sample_fn(
                            model,
                            shape = (1000, args['gene_size']),
                            clip_denoised=args['clip_denoised'],
                            model_kwargs={
                                'z_mod': torch.tensor(z_sem_interp_day_, dtype = torch.float32).to('cuda') 
                            },
                            noise =  noise
        )
        weights_all = weights_all + [day_]* 1000
        interp_all.append(sample_interp_.detach().cpu().numpy())
    interp_all = np.concatenate(interp_all)
    return weights_all, interp_all

# def plot(adata_ctl):
#     ### Define your hexadecimal color codes
#     #Below are cells names
#     #EC series deep red-> orange->yellow

#     hex_colors  = ['red', # EC1
#                 # '#e63d09', # EC2
#                 # '#e66c09', # EC3
#                 # '#e68e09', # EC4
#                 # '#e6ab09', # EC5
#                 # '#e6ce37', # EC6
#                 'green', # Fibroblast1, 
#                 # '#9678f0', # Fibroblast2, 
#                 # '#6a3eed', # Fibroblast3, 
#                 # '#5021de', # Fibroblast4,
#                 # '#8021de', # Fibroblast5, 
#                 # '#af21de', # Fibroblast6, 
#                 # '#c122c9', # Fibroblast7, 
#                 #'#e34f4f', # HSC
#                 'blue'
#                 # Mural1 
#                 # '#21acde', # Mural2
#                 # '#218cde', # Mural3
#                 # '#21de9c', # Mural4
#                 # '#21decb', # Mural5
#                 # '#16c410', # Mural6 
#                 # '#3cb56c', # Mural7
#                 # '#1c702d', # Mural8
#                 # '#63d690' # Mural9 
#                 ]
#     # Convert hexadecimal to RGB
#     rgb_colors = [mcolors.to_rgb(color) for color in hex_colors]

#     # # Create a custom colormap from these colors
#     # cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", rgb_colors)


#     fig, axs = plt.subplots(1,1,figsize=(5,4),dpi=800)
#     for (j,i) in enumerate(np.unique(adata_ctl.obs['type'])):
#         print(i)
#         g=axs.scatter(adata_ctl.obsm['X_umap'][adata_ctl.obs['type']==i, 0], 
#                     adata_ctl.obsm['X_umap'][adata_ctl.obs['type']==i, 1], 
#                     color=hex_colors[j], 
#                     alpha=0.9,s=0.6, 
#                     )
#     #plt.legend(np.unique(subsamp_adata.obs['day']))
#     plt.axis('off')
#     plt.legend()
#     plt.grid(True)
#     print("almost save")
#     plt.savefig("logger_file_fig4/final_clusters.pdf", bbox_inches="tight")

def main():
    
    args = create_argparser()
    args.update({
        'data_dir': '../project_v3/analysis/results/plant_experiment3_predict_days/input',
        'schedule_sampler': 'uniform',
        'lr': 1e-4,
        'weight_decay': 0.0,
        'lr_anneal_steps': 2400,
        'batch_size': 16,
        'microbatch': -1,
        'ema_rate': '0.9999',
        'log_interval': 50,
        'save_interval': 1000,
        'resume_checkpoint': '',
        'use_fp16': False,
        'fp16_scale_growth': 1e-3,
        'gene_size': 39337, 
        'output_dim':39337,
        'num_layers':3,
        'class_cond': False,  
        'use_ddim':True,
        'use_encoder':True,
    })

    logger.configure(dir='logger_file_fig4')
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

    model.load_state_dict(
        dist_util.load_state_dict(args['model_path'])
    )
    model.to(dist_util.dev())
    if args['use_fp16']:
        model.convert_to_fp16()
    model.eval()

    data = sc.read_h5ad(os.path.join(args['data_dir'],'exp3_train_cluster_0_7_11_adata.h5ad'))

    data.obs['Group'].unique()
    print("data Group unique(clusters): ",data.obs['Group'].unique())
    
    weights_all_cluster_7, interp_all_cluster_7 = get_interp(7,0,model,diffusion,data,args, num = 10)
    weights_all_cluster_11, interp_all_cluster_11 = get_interp(11,0,model,diffusion,data,args, num = 10)
    
    interp_all = np.concatenate([interp_all_cluster_7,interp_all_cluster_11])
    weights_all = weights_all_cluster_7 + weights_all_cluster_11
    type_all = ['cluster_7'] * 10000 + ['cluster_11'] * 10000
    interp_all_ann = anndata.AnnData(interp_all)
    interp_all_ann.var_names = data.var_names
    interp_all_ann.obs['label'] = weights_all
    interp_all_ann.obs['type'] = type_all
    # interp_all_ann.obs['train'] = '-1'
    # interp_all_ann.obs['train'][interp_all_ann.obs['label'] != -1]='1'
    interp_all_ann.obs['train'] = '-1'
    mask = interp_all_ann.obs['label'] != -1
    interp_all_ann.obs.loc[mask, 'train'] = '1'
    interp_all_ann.obs['train'] =interp_all_ann.obs['train'].astype('category')
    interp_all_ann.write('interp_all_ann.h5ad')


    # # interp_all_ann.X = np.nan_to_num(interp_all_ann.X, nan=0.0, posinf=0.0, neginf=0.0)
    # sc.pp.pca(interp_all_ann)
    # sc.pp.neighbors(interp_all_ann)
    # sc.tl.umap(interp_all_ann)
    # plot(interp_all_ann)
    
    # sc.tl.embedding_density(interp_all_ann, groupby="type")
    
    
if __name__ == '__main__':
    main()