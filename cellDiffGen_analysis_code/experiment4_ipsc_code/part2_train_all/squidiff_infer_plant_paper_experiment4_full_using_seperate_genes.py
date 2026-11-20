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
from matplotlib import font_manager
font_manager.fontManager.addfont("arial.ttf")
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18,
    "pdf.fonttype": 42,       # Embed TrueType fonts in PDF
})

# def _to_dense_tensor(x, device):
#     arr = x.toarray() if sp.issparse(x) else np.asarray(x)
#     return torch.from_numpy(arr).float().to(device)


# def _resolve_sample_module(project_root):
#     squidiff_dir = os.path.join(project_root, "Squidiff-main")
#     if squidiff_dir not in sys.path:
#         sys.path.insert(0, squidiff_dir)
#     import sample_squidiff
#     return sample_squidiff


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # project_root = os.path.dirname(os.path.abspath(__file__))
    # sample_squidiff = _resolve_sample_module(project_root)

    sampler = sample_squidiff.sampler(
    model_path = 'plant_exp4_squidiff_train_checkpoints_full/model.pt',#'diff_results/model.pt',
    gene_size = 203,
    output_dim = 203,
    use_drug_structure = False
    )

    train_adata = sc.read_h5ad('../project_v3/input/new_train_data.h5ad')
    real_test_adata = sc.read_h5ad('../project_v3/input/test_data.h5ad')
    
    # print(train_adata[train_adata.obs['Group']==1].shape)
    # print(train_adata[train_adata.obs['Group']==2].shape)
    # print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())

    # z_sem_test = sampler.model.encoder(
    #     torch.tensor(real_test_adata.X.toarray()).to('cuda')
    # )
    z_sem_train = sampler.model.encoder(
            torch.tensor(train_adata.X.toarray()).to('cuda')
        )
    z_sem_ctrl = z_sem_train[train_adata.obs['Group'] == 0].mean(axis=0)
    z_sem_PTPN12 = z_sem_train[train_adata.obs['Group'] == 1].mean(axis=0)
    z_sem_ZBTB25 = z_sem_train[train_adata.obs['Group'] == 2].mean(axis=0)


    #plt.xlim([0,4])
    #plt.ylim([0,4])
    # #0520change the original z_sem produce way
    # x = train_adata.X.toarray() if sp.issparse(train_adata.X) else np.asarray(train_adata.X)
    # x = torch.from_numpy(x).float().to("cuda")
    # with torch.no_grad():
    #     z_sem = sampler.model.encoder(x)

    # train_data_day0 = train_adata[train_adata.obs['Group']==0]
    # x0 = train_data_day0.X.toarray() if sp.issparse(train_data_day0.X) else np.asarray(train_data_day0.X)
    # x0 = torch.from_numpy(x0).float().to("cuda")
    # with torch.no_grad():
    #     z_sem_train_0 = sampler.model.encoder(x0)
    
    # direct_z_sem = z_sem[train_adata.obs.Group==12].mean(axis=0) - z_sem[train_adata.obs.Group==0].mean(axis=0)
    # test_adata = sc.read_h5ad('../project_v3/input/plant_paper_test_adata.h5ad')
    # # y = test_adata.X.toarray() if sp.issparse(test_adata.X) else np.asarray(test_adata.X)
    # # y = torch.from_numpy(y).float().to("cuda")

    # # with torch.no_grad():
    # #     z_sem_test = sampler.model.encoder(y)

    r_all = []
    pearsonr_all = []

    for i in range(1):
        final_prediction= sampler.interp_with_direction(z_sem_origin = z_sem_ctrl,
                                        gene_size = 203,
                                        direction = z_sem_PTPN12 + z_sem_ZBTB25 - z_sem_ctrl,
                                        scale = 1,
                                        add_noise_term = False #add noise if for interpolation
                                        )
        
        # final_prediction.detach().cpu().numpy().mean(axis=0).min()

        fig, ax = plt.subplots(1,1,figsize=(4,4),dpi=800)
        pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0)
        real_mean = np.array(real_test_adata.X.mean(axis=0)).ravel()

        # print("2")
        # adata.write('results_add/squidiff_interception_predict_to_day_21.h5ad')
        # print("3")
        plt.scatter(pred_mean,
            real_mean,
            s=10,
            c='blue'
            )
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        limit = np.ceil(max(np.max(pred_mean), np.max(real_mean)))
        print(np.max(pred_mean),np.max(real_mean))
        plt.plot([0, limit], [0, limit], color="black", linestyle=":")
        plt.xticks(np.arange(0, limit + 1, 1))
        plt.yticks(np.arange(0, limit + 1, 1)) 
        plt.xlabel("Predicted mean expression")
        plt.ylabel("Actual mean expression")
        fig.savefig(os.path.join(f"squidiff_exp4_prediction_use_seperate_genes.pdf"), bbox_inches="tight")

        r2_ = r2_score(real_mean,pred_mean)
        pearsonr_, _ = scipy.stats.pearsonr(pred_mean,real_mean)
        #0520change the original cal_metric function
        # r2_, pearsonr_ = sampler.cal_metric(final_prediction, real_test_adata)
        #r2_, pearsonr_ = sampler.cal_metric(sample_interp, real_test_adata)
        r_all.append(['squidiff',i,'day0',r2_])
        pearsonr_all.append(['squidiff',i,'day0',pearsonr_])
    
        
    pd.DataFrame(r_all).to_csv('results_add/squidff_exp4_diff_r_use_seperate_genes.csv')
    pd.DataFrame(pearsonr_all).to_csv('results_add/squidff_exp4_diff_pearsonr_use_seperate_genes.csv')


if __name__ == "__main__":
    # args = build_parser().parse_args()
    run()