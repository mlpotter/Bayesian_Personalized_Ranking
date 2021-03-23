# Reimplementation of BPR: Bayesian Personalized Ranking from Implicit Feedback

Dataset for Netflix may be downloaded from https://www.kaggle.com/netflix-inc/netflix-prize-data

`python BPR_Optimization.py --num_epochs=25 --alpha=0.005 --embed_dim=64 --user_min=10 --item_min=10 --write_filename='' read_filename='data.csv' --batch_size=32 --reg=0.0005 --basepath="archive"`

Argument Descriptions:
	--num_epochs: type=int, help='Number of Training Epochs', default=25
	--alpha: type=float, help='Learning Rate', default=0.005)
	--embed_dim: type=int,help="Size of embedding dimension for matrix factorization",default=64
	--user_min: type=int,help='The approximate minimum number of items each user must have watched',default=10
	--item_min: type=int,help='The approximate minimum number of users each item must have',default=10
	--write_filename: type=str,help='The filename to write all the Netflix data to, and later read',default=""
	--read_filename: type=str,help='The filename to read all the Netflix data from for the Dataframe',default="data.csv"
	--batch_size: type=int,help='The batch size for stochastic gradient descent',default=32
	--reg: type=float,help='The regularization strength on l2 norm',default = 0.0005
	--basepath: type=str,help="The basepath to where the Netflix .txt data files are help",default="archive"

## Match AUC result for Netflix Subsample dataset of ~0.92-0.93