import argparse

def parse_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--in_channels',
        default=1,
        type=int,
        help="Input channels of the net"
    )
    parse.add_argument(
        '--out_channels',
        default=4,
        type=int,
        help="Output channels of the net"
    )
    parse.add_argument(
        '--final_sigmoid',
        default=False,
        type=bool,
        help="Select multi-classification or binary-classification"
    )
    parse.add_argument(
        '--learning_rate',
        default=0.0002,
        type=float,
        help="Initial learning rate"
    )
    parse.add_argument(
        '--epoch',
        default=3,
        type=int,
        help="Number of total epochs to run"
    )
    parse.add_argument(
        '--batch_size',
        default=12,
        type=int,
        help="Batch size"
    )
    parse.add_argument(
        '--num_agents',
        default=6,
        type=int,
        help="The number of agents"
    )
    parse.add_argument(
        '--rounds',
        default=100,
        type=int,
        help="The rounds of the federated learning"
    )
   
    parse.add_argument(
        '--threshold',
        type=float,
        default=0.7
    )

    parse.add_argument('--seed', type=int,  default=2, help='random seed')
    parse.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
    parse.add_argument('--gpu', type=str,  default='4', help='GPU to use')
    args = parse.parse_args()
    return args
