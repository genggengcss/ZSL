# ZSL

### Baseline
(author: Chen Zhuo)  
**DeViSE**: "Devise: A deep visual-semantic embedding model" (pytorch)  
**CONSE**: "Zero-shot learning by convex combination of semantic embeddings" (Matlab)  
**SAE**: "Semantic autoencoder for zero-shot learning" (pytorch)  
**SYNC**: "Synthesized classifiers for zero-shot learning" (Matlab)   
********* 
(author: Geng Yuxia)   
**GCNZ**: "Zero-shot recognition via semantic embeddings and knowledge graphs" (python2 + tensorflow)  
**DGP**: "Rethinking knowledge graph propagation for zero-shot learning" (python3 + pytorch)  
**GAZSL**: "A generative adversarial approach for zero-shot learning from noisy texts" (pytorch)  
**LisGAN**: "Leveraging the invariant side of generative zero-shot learning" (pytorch)  

### Setting
DeViSE, CONSE, SAE and SYNC run with two semantic embeddings (**w2v**: word embedding; **g2v**: trained kg embedding)  
the dimension of **w2v** is 500, and **g2v** is 100.  
w2v download: https://github.com/pujols/zero-shot-learning

GCNZ and DGP 's input is word embedding;  
GAZSL, LisGAN and KG-GAN also run with **w2v** and **g2v**.  


#### Experiment Class Split (for GCNZ, DGP, LisGAN, GAZSL, KG-GAN)
**Exp1**: original animal classes subset (seen:398, unseen:485)  
**Exp9**: "animal" subset ImNet-A (seen:25, unseen:55)  
**Exp10**: "other" subset ImNet-O (seen:10, unseen:25)


### Run Command

#### GCNZ

**construct graph**:  
python io_graph.py --mtr_exp_name Exp9 --exp_name Exp9_2555  

**prepare graph input**:  
python io_train_sample.py --mtr_exp_name Exp9 --exp_name Exp9_2555  

**train**:  
python train_predict_gcn.py --mtr_exp_name Exp9 --exp_name Exp9_2555  

**test:**  
python test_gcn.py --mtr_exp_name Exp9 --exp_name Exp9_2555 --feat 900  
**test (gzsl):**  
python test_gcn.py --mtr_exp_name Exp9 --exp_name Exp9_2555 --feat 900 --gzsl

#### DGP
**prepare graph**:  
python make_induced_graph.py --mtr_exp_name Exp9 --exp_name Exp9_2555

**train**:  
python train_predict_gpm.py --mtr_exp_name Exp9 --exp_name Exp9_2555  

**test**:  
python test_gpm.py --mtr_exp_name Exp9 --exp_name Exp9_2555 --pred 400  

**test (gzsl)**:  
python test_gpm.py --mtr_exp_name Exp9 --exp_name Exp9_2555 --pred 400 --gzsl


#### KG_GAN  
**w2v**: python gan.py --ExpName Exp9 --SemEmbed w2v  
**g2v**: python gan.py --ExpName Exp9 --SemFile g2v.mat --SemSize 100 --NoiseSize 100  

**gzsl**: python gan.py --ExpName Exp9 --SemFile g2v.mat --SemSize 100 --NoiseSize 100 --GZSL
