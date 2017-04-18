


class DDQN_Agent:

    def __init__(self, model, args):
        # Experience Replay
    replay = ExperienceReplay_Options(args.exp_replay_size)
    if args.train_primitives:
        primitive_replay = ExperienceReplay_Options(args.exp_replay_size)

    if args.gpu:
    print("Moving models to GPU.")
    dqn.cuda()
    target_dqn.cuda()

    # Optimizer
optimizer = optim.Adam(dqn.parameters(), lr=args.lr)

def sync_target_network():
    for target, source in zip(target_dqn.parameters(), dqn.parameters()):
        target.data = source.data

        def select_action(state, training=True):
    dqn.eval()
    state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
    q_values = dqn(Variable(state, volatile=True)).cpu().data[0]
    q_values_numpy = q_values.numpy()

    global max_q_value
    global min_q_value
    # Decay it so that it reflects a recentish maximum q value
    max_q_value *= 0.9999
    min_q_value *= 0.9999
    max_q_value = max(max_q_value, np.max(q_values_numpy))
    min_q_value = min(min_q_value, np.min(q_values_numpy))

    if training:
        Q_Values.append(q_values_numpy)

        # Log the q values
        if args.tb:
            # crayon_exp.add_histogram_value("DQN/Q_Values", q_values_numpy.tolist(), tobuild=True, step=T)
            # q_val_dict = {}
            for index in range(args.actions):
                # q_val_dict["DQN/Action_{}_Q_Value".format(index)] = float(q_values_numpy[index])
                if T % args.tb_interval == 0:
                    log_value("DQN/Action_{}_Q_Value".format(index), q_values_numpy[index], step=T)
            # print(q_val_dict)
            # crayon_exp.add_scalar_dict(q_val_dict, step=T)

    if np.random.random() < epsilon:
        action = np.random.randint(low=0, high=args.actions)
    else:
        action = q_values.max(0)[1][0]  # Torch...

    return action, q_values_numpy

    def act(self, state, exp_model):

    def train(self)