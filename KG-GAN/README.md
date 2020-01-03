### Experiment Class Split 
**Exp1**: original animal classes subset (seen:398, unseen:485)

#### Small Subset
**Exp2**: standard split (seen:19, unseen:49)
**Exp3**: proposed split (seen:14, unseen:54)


### Command

#### GCNZ

construct graph:  
python io_graph.py --mtr_exp_name Exp2 --exp_name Exp2_1949

prepare graph input:  
python io_train_sample.py --mtr_exp_name Exp2 --exp_name Exp2_1949

python io_train_sample.py --mtr_exp_name Exp3 --exp_name Exp3_1454 --proposed_split

train:  
python train_predict_gcn.py --mtr_exp_name Exp2 --exp_name Exp2_1949

test: (50 sample)
python test_gcn.py --mtr_exp_name Exp2 --exp_name Exp2_1949 --feat 900 --nsample 50

#### DGP
prepare graph:
python make_induced_graph.py --mtr_exp_name Exp2 --exp_name Exp2_1949

