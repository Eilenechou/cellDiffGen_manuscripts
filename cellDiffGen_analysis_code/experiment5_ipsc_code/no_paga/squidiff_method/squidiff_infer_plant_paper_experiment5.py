import argparse
import os
import sys
import warnings
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import sample_squidiff
from sklearn.metrics import r2_score
import scipy
warnings.filterwarnings("ignore")



def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = sample_squidiff.sampler(
    model_path = 'plant_exp5_squidiff_train_checkpoints/model.pt',
    gene_size = 203,
    output_dim = 203,
    use_drug_structure = False
    )

    train_adata = sc.read_h5ad('../project_v3/input/diff_train.h5ad')
    

    print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())


    # get the day 0 vector from test data
    z_sem = sampler.model.encoder(torch.tensor(train_adata.X).to('cuda'))
    # get the represented differentiation latent vector  
    direct_z_sem = z_sem[train_adata.obs.Group==3].mean(axis=0) - z_sem[train_adata.obs.Group==0].mean(axis=0)
    test_adata = sc.read_h5ad('../project_v3/analysis/results/squidiff_paper_experiment5_figure2/input/diff_test.h5ad')
    z_sem_test_day0 = sampler.model.encoder(torch.tensor(test_adata[test_adata.obs['Group']==0].X).to('cuda'))
    z_sem_test_day3 = sampler.model.encoder(torch.tensor(test_adata[test_adata.obs['Group']==3].X).to('cuda'))

    r_all = []
    pearsonr_all = []

    for i in range(1):
        # sample_interp0 = sampler.pred(z_sem = z_sem_test_day0,
        #                             gene_size = 203,
        #                             )
        sample_interp1 = sampler.interp_with_direction(z_sem_origin = z_sem_test_day0,
                                                    gene_size = 203,
                                                    direction = direct_z_sem,
                                                    scale = 1/3
                                                    )
        sample_interp2 = sampler.interp_with_direction(z_sem_origin = z_sem_test_day0,
                                                    gene_size = 203,
                                                    direction = direct_z_sem,
                                                    scale = 2/3
                                                    )
        sample_interp3 = sampler.interp_with_direction(z_sem_origin = z_sem_test_day0,
                                            gene_size = 203,
                                            direction = direct_z_sem,
                                            scale = 1
                                            )

        adata_1 = anndata.AnnData(sample_interp1.detach().cpu().numpy())
        adata_1.write('results_add/squidiff_exp5_day0_predict_day1.h5ad')
        print("adata_1")
        adata_2 = anndata.AnnData(sample_interp2.detach().cpu().numpy())
        adata_2.write('results_add/squidiff_exp5_day0_predict_day2.h5ad')
        print("adata_2")
        adata_3 = anndata.AnnData(sample_interp3.detach().cpu().numpy())
        adata_3.write('results_add/squidiff_exp5_day0_predict_day3.h5ad')
        print("adata_3")


        z_sem_predict_day1 = sampler.model.encoder(
            torch.tensor(
                adata_1.X.toarray()
                if hasattr(adata_1.X, "toarray")
                else adata_1.X,
                dtype=torch.float32
            ).to("cuda")
        )
        direct_z_sem_new = z_sem[train_adata.obs.Group==3].mean(axis=0) - z_sem_predict_day1.mean(axis=0)
        
        sample_interp2_new = sampler.interp_with_direction(z_sem_origin = z_sem_predict_day1,
                                    gene_size = 203,
                                    direction = direct_z_sem_new,
                                    scale = 1/2
                                    )   

        adata_2_new = anndata.AnnData(sample_interp2_new.detach().cpu().numpy())                        
        print("2")
        adata_2_new.write('results_add/squidiff_exp5_predictedday1_to_predict_day2.h5ad')
        print("3")
        # r2_, pearsonr_ = sampler.cal_metric(sample_interp0, test_adata[test_adata.obs['Group']==0])
        # r_all.append(['squidiff',i,'day0',r2_])
        # pearsonr_all.append(['squidiff',i,'day0',pearsonr_])
        # r2_, pearsonr_ = sampler.cal_metric(sample_interp1, test_adata[test_adata.obs['Group']==1])
        # r_all.append(['squidiff',i,'day1',r2_])
        # pearsonr_all.append(['squidiff',i,'day1',pearsonr_])
        # r2_, pearsonr_ = sampler.cal_metric(sample_interp2,  test_adata[test_adata.obs['Group']==2])
        # r_all.append(['squidiff',i,'day2',r2_])
        # pearsonr_all.append(['squidiff',i,'day2',pearsonr_])
        # r2_, pearsonr_ = sampler.cal_metric(sample_interp3, test_adata[test_adata.obs['Group']==3])
        # r_all.append(['squidiff',i,'day3',r2_])
        # pearsonr_all.append(['squidiff',i,'day3',pearsonr_])

    #     plt.subplots(1,1,figsize=(4,4),dpi=800)
    #     plt.scatter(sample_interp0.detach().cpu().mean(axis=0),
    #                 test_adata[test_adata.obs['Group']==0].X.mean(axis=0),
    #                 s=4,
    #                 c='#c29523'
    #             )
    #     plt.xlim([0,10])
    #     plt.ylim([0,10])

    #     sampler.cal_metric(sample_interp0, test_adata[test_adata.obs['Group']==0])
    #     fig.savefig(os.path.join(f"prediction_squidiff_exp5_{i}.pdf"), bbox_inches="tight")


    #     plt.subplots(1,1,figsize=(2,2),dpi=800)
    #     plt.scatter(sample_interp1.detach().cpu().mean(axis=0),
    #                 test_adata[test_adata.obs['Group']==1].X.mean(axis=0),
    #                 s=4,
    #                 c='#c29523'
    #             )
    #     plt.xlim([0,10])
    #     plt.ylim([0,10])

    #     sampler.cal_metric(sample_interp1, test_adata[test_adata.obs['Group']==1])
    #     fig.savefig(os.path.join(f"prediction_squidiff_exp5_{i}.pdf"), bbox_inches="tight")

    #     plt.subplots(1,1,figsize=(2,2),dpi=800)
    #     plt.scatter(sample_interp2.detach().cpu().mean(axis=0),
    #                 test_adata[test_adata.obs['Group']==2].X.mean(axis=0),
    #                 s=4,
    #                 c='#c29523'
    #             )
    #     plt.xlim([0,10])
    #     plt.ylim([0,10])

    #     sampler.cal_metric(sample_interp2, test_adata[test_adata.obs['Group']==2])
    #     fig.savefig(os.path.join(f"prediction_squidiff_exp5_{i}.pdf"), bbox_inches="tight")

    #     plt.subplots(1,1,figsize=(2,2),dpi=800)
    #     plt.scatter(sample_interp3.detach().cpu().mean(axis=0),
    #                 test_adata[test_adata.obs['Group']==3].X.mean(axis=0),
    #                 s=4,
    #                 c='#c29523'
    #             )
    #     plt.xlim([0,10])
    #     plt.ylim([0,10])

    #     sampler.cal_metric(sample_interp3, test_adata[test_adata.obs['Group']==3])
    #     fig.savefig(os.path.join(f"prediction_squidiff_exp5_{i}.pdf"), bbox_inches="tight")

            
    # pd.DataFrame(r_all).to_csv('results_add/fig2_diff_r_all_new.csv')
    # pd.DataFrame(pearsonr_all).to_csv('results_add/fig2_diff_pearsonr_all_new.csv')



if __name__ == "__main__":
    # args = build_parser().parse_args()
    run()