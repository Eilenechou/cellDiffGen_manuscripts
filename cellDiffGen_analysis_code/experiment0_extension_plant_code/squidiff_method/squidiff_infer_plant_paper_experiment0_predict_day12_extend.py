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
    model_path = 'plant_exp0_squidiff_train_checkpoints_predict_day12_top200/model.pt',#'diff_results/model.pt',
    gene_size = 200,
    output_dim = 200,
    use_drug_structure = False
    )

    train_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_train_for_day12_adata_gene_top200.h5ad')
    
    # print(train_adata[train_adata.obs['Group']==12].shape)
    # print(train_adata[train_adata.obs['Group']==21].shape)
    print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())

    #0520change the original z_sem produce way
    x = train_adata.X.toarray() if sp.issparse(train_adata.X) else np.asarray(train_adata.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem = sampler.model.encoder(x)

    train_data_day = train_adata[train_adata.obs['Group']==21]
    x = train_data_day.X.toarray() if sp.issparse(train_data_day.X) else np.asarray(train_data_day.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem_train_21 = sampler.model.encoder(x)
    
    direct_z_sem = z_sem[train_adata.obs.Group==0].mean(axis=0) - z_sem[train_adata.obs.Group==21].mean(axis=0)
    test_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_real_test_day12_data_gene_top200.h5ad')


    r_all = []
    pearsonr_all = []

    for i in range(1):
        final_prediction = sampler.interp_with_direction(z_sem_origin = z_sem_train_21,
                                                    gene_size = 200,
                                                    direction = direct_z_sem,
                                                    scale = 12-21/-21
                                                    )
        adata = anndata.AnnData(final_prediction.detach().cpu().numpy())
        print("2")
        adata.write('results_add/squidiff_exp0_interception_predict_day_12_top200.h5ad')
        print("3")
        pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0)
        real_mean = np.asarray(test_adata.X.mean(axis=0)).ravel()
        
        r2_ = r2_score(pred_mean, real_mean)
        pearsonr_, _ = scipy.stats.pearsonr(pred_mean, real_mean)
        #0520change the original cal_metric function
        #r2_, pearsonr_ = sampler.cal_metric(sample_interp, test_adata)
        r_all.append(['squidiff',i,'predict day0',r2_])
        pearsonr_all.append(['squidiff',i,'predict day0',pearsonr_])
    
        
    pd.DataFrame(r_all).to_csv('results_add/plant_exp0_predict_day12_diff_r_top200.csv')
    pd.DataFrame(pearsonr_all).to_csv('results_add/plant_exp0_predict_day12_diff_pearsonr_top200.csv')

def run_2():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = sample_squidiff.sampler(
    model_path = 'plant_exp0_squidiff_train_checkpoints_predict_day12_top500/model.pt',#'diff_results/model.pt',
    gene_size = 500,
    output_dim = 500,
    use_drug_structure = False
    )

    train_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_train_for_day12_adata_gene_top500.h5ad')
    
    # print(train_adata[train_adata.obs['Group']==12].shape)
    # print(train_adata[train_adata.obs['Group']==21].shape)
    print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())

    #0520change the original z_sem produce way
    x = train_adata.X.toarray() if sp.issparse(train_adata.X) else np.asarray(train_adata.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem = sampler.model.encoder(x)

    train_data_day = train_adata[train_adata.obs['Group']==21]
    x = train_data_day.X.toarray() if sp.issparse(train_data_day.X) else np.asarray(train_data_day.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem_train_21 = sampler.model.encoder(x)
    
    direct_z_sem = z_sem[train_adata.obs.Group==0].mean(axis=0) - z_sem[train_adata.obs.Group==21].mean(axis=0)
    test_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_real_test_day12_data_gene_top500.h5ad')


    r_all = []
    pearsonr_all = []

    for i in range(1):
        final_prediction = sampler.interp_with_direction(z_sem_origin = z_sem_train_21,
                                                    gene_size = 500,
                                                    direction = direct_z_sem,
                                                    scale = 12-21/-21
                                                    )
        adata = anndata.AnnData(final_prediction.detach().cpu().numpy())
        print("2")
        adata.write('results_add/squidiff_exp0_interception_predict_day_12_top500.h5ad')
        print("3")
        pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0)
        real_mean = np.asarray(test_adata.X.mean(axis=0)).ravel()
        
        r2_ = r2_score(pred_mean, real_mean)
        pearsonr_, _ = scipy.stats.pearsonr(pred_mean, real_mean)
        #0520change the original cal_metric function
        #r2_, pearsonr_ = sampler.cal_metric(sample_interp, test_adata)
        r_all.append(['squidiff',i,'predict day0',r2_])
        pearsonr_all.append(['squidiff',i,'predict day0',pearsonr_])
    
        
    pd.DataFrame(r_all).to_csv('results_add/plant_exp0_predict_day12_diff_r_top500.csv')
    pd.DataFrame(pearsonr_all).to_csv('results_add/plant_exp0_predict_day12_diff_pearsonr_top500.csv')


def run_3():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = sample_squidiff.sampler(
    model_path = 'plant_exp0_squidiff_train_checkpoints_predict_day12_top2000/model.pt',#'diff_results/model.pt',
    gene_size = 2000,
    output_dim = 2000,
    use_drug_structure = False
    )

    train_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_train_for_day12_adata_gene_top2000.h5ad')
    
    # print(train_adata[train_adata.obs['Group']==12].shape)
    # print(train_adata[train_adata.obs['Group']==21].shape)
    print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())

    #0520change the original z_sem produce way
    x = train_adata.X.toarray() if sp.issparse(train_adata.X) else np.asarray(train_adata.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem = sampler.model.encoder(x)

    train_data_day = train_adata[train_adata.obs['Group']==21]
    x = train_data_day.X.toarray() if sp.issparse(train_data_day.X) else np.asarray(train_data_day.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem_train_21 = sampler.model.encoder(x)
    
    direct_z_sem = z_sem[train_adata.obs.Group==0].mean(axis=0) - z_sem[train_adata.obs.Group==21].mean(axis=0)
    test_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_real_test_day12_data_gene_top2000.h5ad')


    r_all = []
    pearsonr_all = []

    for i in range(1):
        final_prediction = sampler.interp_with_direction(z_sem_origin = z_sem_train_21,
                                                    gene_size = 2000,
                                                    direction = direct_z_sem,
                                                    scale = 12-21/-21
                                                    )
        adata = anndata.AnnData(final_prediction.detach().cpu().numpy())
        print("2")
        adata.write('results_add/squidiff_exp0_interception_predict_day_12_top2000.h5ad')
        print("3")
        pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0)
        real_mean = np.asarray(test_adata.X.mean(axis=0)).ravel()
        
        r2_ = r2_score(pred_mean, real_mean)
        pearsonr_, _ = scipy.stats.pearsonr(pred_mean, real_mean)
        #0520change the original cal_metric function
        #r2_, pearsonr_ = sampler.cal_metric(sample_interp, test_adata)
        r_all.append(['squidiff',i,'predict day0',r2_])
        pearsonr_all.append(['squidiff',i,'predict day0',pearsonr_])
    
        
    pd.DataFrame(r_all).to_csv('results_add/plant_exp0_predict_day12_diff_r_top2000.csv')
    pd.DataFrame(pearsonr_all).to_csv('results_add/plant_exp0_predict_day12_diff_pearsonr_top2000.csv')

def run_4():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = sample_squidiff.sampler(
    model_path = 'plant_exp0_squidiff_train_checkpoints_predict_day12_top5000/model.pt',#'diff_results/model.pt',
    gene_size = 5000,
    output_dim = 5000,
    use_drug_structure = False
    )

    train_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_train_for_day12_adata_gene_top5000.h5ad')
    
    # print(train_adata[train_adata.obs['Group']==12].shape)
    # print(train_adata[train_adata.obs['Group']==21].shape)
    print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())

    #0520change the original z_sem produce way
    x = train_adata.X.toarray() if sp.issparse(train_adata.X) else np.asarray(train_adata.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem = sampler.model.encoder(x)

    train_data_day = train_adata[train_adata.obs['Group']==21]
    x = train_data_day.X.toarray() if sp.issparse(train_data_day.X) else np.asarray(train_data_day.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem_train_21 = sampler.model.encoder(x)
    
    direct_z_sem = z_sem[train_adata.obs.Group==0].mean(axis=0) - z_sem[train_adata.obs.Group==21].mean(axis=0)
    test_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_real_test_day12_data_gene_top5000.h5ad')


    r_all = []
    pearsonr_all = []

    for i in range(1):
        final_prediction = sampler.interp_with_direction(z_sem_origin = z_sem_train_21,
                                                    gene_size = 5000,
                                                    direction = direct_z_sem,
                                                    scale = 12-21/-21
                                                    )
        adata = anndata.AnnData(final_prediction.detach().cpu().numpy())
        print("2")
        adata.write('results_add/squidiff_exp0_interception_predict_day_12_top5000.h5ad')
        print("3")
        pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0)
        real_mean = np.asarray(test_adata.X.mean(axis=0)).ravel()
        
        r2_ = r2_score(pred_mean, real_mean)
        pearsonr_, _ = scipy.stats.pearsonr(pred_mean, real_mean)
        #0520change the original cal_metric function
        #r2_, pearsonr_ = sampler.cal_metric(sample_interp, test_adata)
        r_all.append(['squidiff',i,'predict day0',r2_])
        pearsonr_all.append(['squidiff',i,'predict day0',pearsonr_])
    
        
    pd.DataFrame(r_all).to_csv('results_add/plant_exp0_predict_day12_diff_r_top5000.csv')
    pd.DataFrame(pearsonr_all).to_csv('results_add/plant_exp0_predict_day12_diff_pearsonr_top5000.csv')


def run_5():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = sample_squidiff.sampler(
    model_path = 'plant_exp0_squidiff_train_checkpoints_predict_day12_top10000/model.pt',#'diff_results/model.pt',
    gene_size = 10000,
    output_dim = 10000,
    use_drug_structure = False
    )

    train_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_train_for_day12_adata_gene_top10000.h5ad')
    
    # print(train_adata[train_adata.obs['Group']==12].shape)
    # print(train_adata[train_adata.obs['Group']==21].shape)
    print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())

    #0520change the original z_sem produce way
    x = train_adata.X.toarray() if sp.issparse(train_adata.X) else np.asarray(train_adata.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem = sampler.model.encoder(x)

    train_data_day = train_adata[train_adata.obs['Group']==21]
    x = train_data_day.X.toarray() if sp.issparse(train_data_day.X) else np.asarray(train_data_day.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem_train_21 = sampler.model.encoder(x)
    
    direct_z_sem = z_sem[train_adata.obs.Group==0].mean(axis=0) - z_sem[train_adata.obs.Group==21].mean(axis=0)
    test_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_real_test_day12_data_gene_top10000.h5ad')


    r_all = []
    pearsonr_all = []

    for i in range(1):
        final_prediction = sampler.interp_with_direction(z_sem_origin = z_sem_train_21,
                                                    gene_size = 10000,
                                                    direction = direct_z_sem,
                                                    scale = 12-21/-21
                                                    )
        adata = anndata.AnnData(final_prediction.detach().cpu().numpy())
        print("2")
        adata.write('results_add/squidiff_exp0_interception_predict_day_12_top10000.h5ad')
        print("3")
        pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0)
        real_mean = np.asarray(test_adata.X.mean(axis=0)).ravel()
        
        r2_ = r2_score(pred_mean, real_mean)
        pearsonr_, _ = scipy.stats.pearsonr(pred_mean, real_mean)
        #0520change the original cal_metric function
        #r2_, pearsonr_ = sampler.cal_metric(sample_interp, test_adata)
        r_all.append(['squidiff',i,'predict day0',r2_])
        pearsonr_all.append(['squidiff',i,'predict day0',pearsonr_])
    
        
    pd.DataFrame(r_all).to_csv('results_add/plant_exp0_predict_day12_diff_r_top10000.csv')
    pd.DataFrame(pearsonr_all).to_csv('results_add/plant_exp0_predict_day12_diff_pearsonr_top10000.csv')

def run_6():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = sample_squidiff.sampler(
    model_path = 'plant_exp0_squidiff_train_checkpoints_predict_day12_top20000/model.pt',#'diff_results/model.pt',
    gene_size = 20000,
    output_dim = 20000,
    use_drug_structure = False
    )

    train_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_train_for_day12_adata_gene_top20000.h5ad')
    
    # print(train_adata[train_adata.obs['Group']==12].shape)
    # print(train_adata[train_adata.obs['Group']==21].shape)
    print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())

    #0520change the original z_sem produce way
    x = train_adata.X.toarray() if sp.issparse(train_adata.X) else np.asarray(train_adata.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem = sampler.model.encoder(x)

    train_data_day = train_adata[train_adata.obs['Group']==21]
    x = train_data_day.X.toarray() if sp.issparse(train_data_day.X) else np.asarray(train_data_day.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem_train_21 = sampler.model.encoder(x)
    
    direct_z_sem = z_sem[train_adata.obs.Group==0].mean(axis=0) - z_sem[train_adata.obs.Group==21].mean(axis=0)
    test_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_real_test_day12_data_gene_top20000.h5ad')


    r_all = []
    pearsonr_all = []

    for i in range(1):
        final_prediction = sampler.interp_with_direction(z_sem_origin = z_sem_train_21,
                                                    gene_size = 20000,
                                                    direction = direct_z_sem,
                                                    scale = 12-21/-21
                                                    )
        adata = anndata.AnnData(final_prediction.detach().cpu().numpy())
        print("2")
        adata.write('results_add/squidiff_exp0_interception_predict_day_12_top20000.h5ad')
        print("3")
        pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0)
        real_mean = np.asarray(test_adata.X.mean(axis=0)).ravel()
        
        r2_ = r2_score(pred_mean, real_mean)
        pearsonr_, _ = scipy.stats.pearsonr(pred_mean, real_mean)
        #0520change the original cal_metric function
        #r2_, pearsonr_ = sampler.cal_metric(sample_interp, test_adata)
        r_all.append(['squidiff',i,'predict day0',r2_])
        pearsonr_all.append(['squidiff',i,'predict day0',pearsonr_])
    
        
    pd.DataFrame(r_all).to_csv('results_add/plant_exp0_predict_day12_diff_r_top20000.csv')
    pd.DataFrame(pearsonr_all).to_csv('results_add/plant_exp0_predict_day12_diff_pearsonr_top20000.csv')


def run_7():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = sample_squidiff.sampler(
    model_path = 'plant_exp0_squidiff_train_checkpoints_predict_day12_top30000/model.pt',#'diff_results/model.pt',
    gene_size = 30000,
    output_dim = 30000,
    use_drug_structure = False
    )

    train_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_train_for_day12_adata_gene_top30000.h5ad')
    
    # print(train_adata[train_adata.obs['Group']==12].shape)
    # print(train_adata[train_adata.obs['Group']==21].shape)
    print(train_adata)
    train_adata.obs['Group'].unique()
    print(train_adata.obs['Group'].unique().tolist())

    #0520change the original z_sem produce way
    x = train_adata.X.toarray() if sp.issparse(train_adata.X) else np.asarray(train_adata.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem = sampler.model.encoder(x)

    train_data_day = train_adata[train_adata.obs['Group']==21]
    x = train_data_day.X.toarray() if sp.issparse(train_data_day.X) else np.asarray(train_data_day.X)
    x = torch.from_numpy(x).float().to("cuda")
    with torch.no_grad():
        z_sem_train_21 = sampler.model.encoder(x)
    
    direct_z_sem = z_sem[train_adata.obs.Group==0].mean(axis=0) - z_sem[train_adata.obs.Group==21].mean(axis=0)
    test_adata = sc.read_h5ad('../project_v3/analysis/results/plant_experiment0_extension/predict_day12/input/plant_real_test_day12_data_gene_top30000.h5ad')


    r_all = []
    pearsonr_all = []

    for i in range(1):
        final_prediction = sampler.interp_with_direction(z_sem_origin = z_sem_train_21,
                                                    gene_size = 30000,
                                                    direction = direct_z_sem,
                                                    scale = 12-21/-21
                                                    )
        adata = anndata.AnnData(final_prediction.detach().cpu().numpy())
        print("2")
        adata.write('results_add/squidiff_exp0_interception_predict_day_12_top30000.h5ad')
        print("3")
        pred_mean = final_prediction.detach().cpu().numpy().mean(axis=0)
        real_mean = np.asarray(test_adata.X.mean(axis=0)).ravel()
        
        r2_ = r2_score(pred_mean, real_mean)
        pearsonr_, _ = scipy.stats.pearsonr(pred_mean, real_mean)
        #0520change the original cal_metric function
        #r2_, pearsonr_ = sampler.cal_metric(sample_interp, test_adata)
        r_all.append(['squidiff',i,'predict day0',r2_])
        pearsonr_all.append(['squidiff',i,'predict day0',pearsonr_])
    
        
    pd.DataFrame(r_all).to_csv('results_add/plant_exp0_predict_day12_diff_r_top30000.csv')
    pd.DataFrame(pearsonr_all).to_csv('results_add/plant_exp0_predict_day12_diff_pearsonr_top30000.csv')

if __name__ == "__main__":
    # args = build_parser().parse_args()
    run()
    run_2()
    run_3()
    run_4()
    run_5()
    run_6()
    run_7()