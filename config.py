import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--seed", type=int, default=19260817)
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--dataset_root", type=str, default="/home/MaCS/fedDM/datasets/torchvision")
parser.add_argument("--split_file", type=str, default="")
parser.add_argument("--dataset", type=str, default='CIFAR10')
parser.add_argument("--client_num", type=int, default=50)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--partition_method", type=str, default='part', choices=['part', 'only'],
                    help='Data partition method: part (class-based + dirichlet) or only (direct dirichlet)')

parser.add_argument("--algorithm", type=str, default="fedDM", choices=['fedDM', 'fedprotoDM', 'trueprotoDM'],
                    help='Choose algorithm: fedDM (original), fedprotoDM (server-side synthesis), or trueprotoDM (true prototype-based synthesis)')
parser.add_argument("--model", type=str, default="ConvNet")
parser.add_argument("--communication_rounds", type=int, default=20)
parser.add_argument("--join_ratio", type=float, default=1.0)

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--model_epochs", type=int, default=500)

parser.add_argument("--ipc", type=int, default=10)
parser.add_argument("--rho", type=int, default=5)
parser.add_argument("--dc_iterations", type=int, default=1000)
parser.add_argument("--dc_batch_size", type=int, default=256)
parser.add_argument("--image_lr", type=float, default=1)

parser.add_argument("--init_method", type=str, default="real_sample", choices=['real_sample', 'random'],
                    help='Initialization method for synthetic images: real_sample (use real samples) or random (use random noise)')
parser.add_argument("--eval_gap", type=int, default=1)

