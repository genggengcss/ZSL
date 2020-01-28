# ZSL

### Baseline
GCNZ: "Zero-shot recognition via semantic embeddings and knowledge graphs"
DGP: "Rethinking knowledge graph propagation for zero-shot learning"
GAZSL: "A generative adversarial approach for zero-shot learning from noisy texts"
LisGAN: "Leveraging the invariant side of generative zero-shot learning"

### Experiment Class Split 
**Exp1**: original animal classes subset (seen:398, unseen:485)

#### Small Subset
**Exp2**: standard split (seen:19, unseen:49)  
**Exp3**: proposed split (seen:14, unseen:54)


### Run Command

#### GCNZ

**construct graph**:  
python io_graph.py --mtr_exp_name Exp2 --exp_name Exp2_1949  

**prepare graph input**:  
python io_train_sample.py --mtr_exp_name Exp2 --exp_name Exp2_1949  
python io_train_sample.py --mtr_exp_name Exp3 --exp_name Exp3_1454 --proposed_split

**train**:  
python train_predict_gcn.py --mtr_exp_name Exp2 --exp_name Exp2_1949  

**test (50 sample):**  
python test_gcn.py --mtr_exp_name Exp2 --exp_name Exp2_1949 --feat 900 --nsample 50  
python test_gcn.py --mtr_exp_name Exp3 --exp_name Exp3_1454 --feat 850 --nsample 50 --proposed_split  
**test (gzsl):**  
python test_gcn.py --mtr_exp_name Exp2 --exp_name Exp2_1949 --feat 900 --nsample 50 --gzsl

#### DGP
**prepare graph**:  
python make_induced_graph.py --mtr_exp_name Exp2 --exp_name Exp2_1949

**train**:  
python train_predict_gpm.py --mtr_exp_name Exp2 --exp_name Exp2_1949  
python train_predict_gpm.py --mtr_exp_name Exp3 --exp_name Exp3_1454 --proposed_split

**test**:  
python test_gpm.py --mtr_exp_name Exp2 --exp_name Exp2_1949 --pred 400 --nsample 50  
python test_gpm.py --mtr_exp_name Exp3 --exp_name Exp3_1454 --pred 300 --nsample 50 --proposed_split

**test (gzsl)**:  
python test_gpm.py --mtr_exp_name Exp2 --exp_name Exp2_1949 --pred 400 --nsample 50 --gzsl


#### KG_GAN  
**w2v**: python lisgan.py --ExpName Exp2 --SemEmbed w2v  
**w2v**: python lisgan.py --ExpName Exp3 --SemEmbed w2v --ProposedSplit  
**n2v**: python lisgan.py --ExpName Exp2 --SemFile n2v.mat --SemSize 100 --NoiseSize 50  

**gzsl**: python lisgan.py --ExpName Exp2 --SemFile n2v.mat --SemSize 100 --NoiseSize 50 --GZSL
