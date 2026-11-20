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
                                        model_path = 'plant_exp1_squidiff_train_checkpoints/model.pt',#'diff_results/model.pt',
                                        gene_size = 39337,
                                        output_dim = 39337,
                                        use_drug_structure = False
                                        )

    train_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment1_day12_to_day21/inputs/plant_experiment1_train_adata.h5ad')
    
    print(train_adata[train_adata.obs['Group']==12].shape)
    print(train_adata[train_adata.obs['Group']==21].shape)
    print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())

    #0520change the original z_sem produce way
    x = train_adata.X.toarray() if sp.issparse(train_adata.X) else np.asarray(train_adata.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        print("go to this part, do not need to rerun again.")
        z_sem = sampler.model.encoder(x)

    test_adata_day12 = sc.read_h5ad('../project_v3/analysis/results/plant_experiment1_day12_to_day21/inputs/plant_paper_test_day12_adata.h5ad')
    X_12 = test_adata_day12.X.toarray() if sp.issparse(test_adata_day12.X) else test_adata_day12.X
    z_sem_test_day12 = sampler.model.encoder(torch.tensor(X_12, dtype=torch.float32).to("cuda"))
    # since test_adata_day12 is sparse, different than the squidiff original dataset.
    # z_sem_test_day12 = sampler.model.encoder(torch.tensor(test_adata_day12.X).to('cuda'))
    direct_z_sem = z_sem[train_adata.obs.Group==21].mean(axis=0) - z_sem[train_adata.obs.Group==12].mean(axis=0)

    r_all = []
    pearsonr_all = []

    # for i in range(1):
    test_adata_day21 = sc.read_h5ad('../project_v3/analysis/results/plant_experiment1_day12_to_day21/inputs/plant_paper_test_day21_adata.h5ad')
    test_adata_day21.X = test_adata_day21.X.toarray() if sp.issparse(test_adata_day21.X) else test_adata_day21.X
    final_prediction = sampler.interp_with_direction(z_sem_origin = z_sem_test_day12,
                                                gene_size = 39337,
                                                direction = direct_z_sem,
                                                scale = 1
                                                )
    adata_whole = anndata.AnnData(final_prediction.detach().cpu().numpy())
    print("2")
    adata_whole.write('logger_file_plant_exp1_1e5/squidiff_plant_exp1_interception_predict_to_day_21.h5ad')
    print("3")
    r2_, pearsonr_ = sampler.cal_metric(final_prediction, test_adata_day21)
    # pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0).ravel()
    # real_mean = np.asarray(test_adata_day12.X.mean(axis=0)).ravel()
    # r2_ = r2_score(real_mean, pred_mean)
    # pearsonr_, _ = scipy.stats.pearsonr(real_mean, pred_mean)
    #0520change the original cal_metric function
    #r2_, pearsonr_ = sampler.cal_metric(sample_interp, test_adata)
    # r_all.append(['squidiff',"whole",'day21',r2_])
    # pearsonr_all.append(['squidiff',"whole",'day21',pearsonr_])

    r_all.append(['squidiff', "whole", 'predict day21 from test data day12', 'predict number:', adata_whole.shape[0], 'test_data_number:',test_adata_day21.shape[0], 'r2_score:',r2_])
    pearsonr_all.append(['squidiff', "whole", 'predict day21 from test data day12', 'predict number:', adata_whole.shape[0], 'test_data_number:',test_adata_day21.shape[0], 'pearson_score:',pearsonr_])

    
# print(len(real_data.obs["Cluster"].unique()))

    for i in range(15):
        if i == 11:
            continue
        real_data_test_12 = sc.read_h5ad(f"../project_v3/analysis/results/plant_experiment1_day12_to_day21/inputs/test_day12_plant_experiment1_cluster_{i}.h5ad")
        X_12_cluster = real_data_test_12.X.toarray() if sp.issparse(real_data_test_12.X) else real_data_test_12.X
        z_sem_test_day12_cluster_i = sampler.model.encoder(torch.tensor(X_12_cluster, dtype=torch.float32).to("cuda"))

        real_data_cluster = sc.read_h5ad(f"../project_v3/analysis/results/plant_experiment1_day12_to_day21/inputs/test_day21_plant_experiment1_cluster_{i}.h5ad")
        real_data_cluster.X = real_data_cluster.X.toarray() if sp.issparse(real_data_cluster.X) else real_data_cluster.X

        final_prediction_cluster_i = sampler.interp_with_direction(z_sem_origin = z_sem_test_day12_cluster_i,
                                                    gene_size = 39337,
                                                    direction = direct_z_sem,
                                                    scale = 1
                                                    )
        adata = anndata.AnnData(final_prediction_cluster_i.detach().cpu().numpy())
        print("2")
        adata.write(f'logger_file_plant_exp1_1e5/squidiff_plant_exp1_interception_predict_to_day_21_cluster_{i}.h5ad')
        print("3")
        # pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0).ravel()
        # real_mean = np.asarray(test_adata_day21.X.mean(axis=0)).ravel()
        # r2_ = r2_score(final_prediction, pred_mean)
        # pearsonr_, _ = scipy.stats.pearsonr(real_mean, pred_mean)
        #0520change the original cal_metric function
        #r2_, pearsonr_ = sampler.cal_metric(sample_interp, test_adata)
        r2_, pearsonr_ = sampler.cal_metric(final_prediction_cluster_i, real_data_cluster)
        
        # r_all.append(['squidiff_cluster_id',i,'day21',r2_])
        # pearsonr_all.append(['squidiff_cluster_id',i,'day21',pearsonr_])
        r_all.append(['squidiff', f"cluster{i}", 'predict day21 from test data day12', 'predict number:', adata.shape[0], 'test_data_number:',real_data_cluster.shape[0], 'r2_score:',r2_])
        pearsonr_all.append(['squidiff', f"cluster{i}", 'predict day21 from test data day12', 'predict number:', adata.shape[0], 'test_data_number:',real_data_cluster.shape[0], 'pearson_score:',pearsonr_])

            
    pd.DataFrame(r_all).to_csv('logger_file_plant_exp1_1e5/squidiff_plant_exp1_diff_r2.csv')
    pd.DataFrame(pearsonr_all).to_csv('logger_file_plant_exp1_1e5/squidiff_plant_exp1_diff_pearsonr.csv')



if __name__ == "__main__":
    # args = build_parser().parse_args()
    run()