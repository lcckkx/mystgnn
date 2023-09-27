# mystgnn
1 genarate label
python main.py --simulate --env (env_name) --data_dir (data_name) (--edge_fusion) (--act)
Simulations are made to generate training data at ./envs/data/env_name/data_name/.
2 train
python main.py --train --env (env_name) --data_dir (data_name)  --model_dir (model_name) (--edge_fusion) (--act) (--conv GAT) (--recurrent Conv1D) (--batch_size 64) (--epochs 20000) (--if_flood) (--norm) (--resnet) (--seq_in 10) (--seq_out 10)
The model structure is built and trained with data at data_dir for epochs. Details of the model and training parameters refer to config.yaml. The trained model and training loss logging are saved at ./model/env_name/model_name/.
3 testing
python main.py --test --env (env_name) --model_dir (model_name) --result_dir (result_name) (--edge_fusion) (--act) (--conv GAT) (--recurrent Conv1D) (--if_flood) (--norm) (--resnet) (--seq_in 10) (--seq_out 10)
The model is loaded to emulate the drainage network in various rainfalls. Details of the model and testing parameters refer to config.yaml and parser func at main.py. The testing states, performance (perfs), settings and prediction results of each rainfall are saved at ./result/env_name/result_name/.
