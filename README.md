# [Under review] Improving Angular Embeddings in GCN for Recommendations

We propose AIM (Angular embedding Improvement and Magnitude balancing), a simple yet effective method that enhances angular embedding and mitigates magnitude imbalance.

# Requirements
python 3.8.18, cuda 11.8, and the following installations:
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-geometric
pip install six
pip install pandas
```

# Run
Instead of **[dataset]**, substitute **Amazon-CD**, **Gowalla**, **Yelp** to run the code.
##### DNA-LightGCN
```
python main_DNA-LightGCN.py --dataset [dataset]
```
