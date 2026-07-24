import subprocess
import anndata
import tqdm

DATA_FOLDER = "../CoPhaser/data/"


adata = anndata.read_h5ad(
    f"{DATA_FOLDER}cellcycle_maxine/VASA_preprocesseed_moments.h5ad"
)
command = [
    "python",
    "DeepCycle.py",
    "--input_adata",
    f"{DATA_FOLDER}cellcycle_maxine/tmp_VASA_preprocesseed_moments_celltype.h5ad",
    "--gene_list",
    "go_annotation/GO_cell_cycle_annotation_mouse.txt",
    "--base_gene",
    "Top2a",
    "--expression_threshold",
    "1",
    "--gpu",
    "--hotelling",
    "--output_adata",
]

for celltype in tqdm.tqdm(adata.obs["Celltype"].unique()):
    adata_celltype = adata[adata.obs["Celltype"] == celltype]
    adata_celltype.write_h5ad(
        f"{DATA_FOLDER}cellcycle_maxine/tmp_VASA_preprocesseed_moments_celltype.h5ad"
    )
    output = [
        f"/home/maxine/Documents/paychere/CoPhaser/paper/code/figure_2/vasa/deepcycle_all_genes_res/{celltype}.h5ad"
    ]
    try:
        subprocess.run(
            command + output,
            check=True,
            cwd="/home/maxine/Documents/paychere/DeepCycle",
        )
    except subprocess.CalledProcessError as e:
        # write the error message to a log file
        with open(
            "/home/maxine/Documents/paychere/CoPhaser/paper/code/figure_2/vasa/deepcycle_all_genes_res/error_log.txt",
            "a",
        ) as f:
            f.write(
                f"DeepCycle failed with exit code {e.returncode} for celltype {celltype}\n"
            )
