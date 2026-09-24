import torch

from crcm.config import _settings
from crcm.model import init_model, test_model, train_crcm


def main():
    torch.set_num_threads(1)
    params, reward_params = _settings()
    (
        net,
        _,
        traj_train,
        traj_test,
        traj_true,
        flow_train,
        state_feature,
        action_feature,
    ) = init_model(params, reward_params)
    reward_model, gamma, _ = train_crcm(
        net,
        traj_train,
        flow_train,
        state_feature,
        action_feature,
        params,
        reward_params,
    )
    results = test_model(
        net,
        traj_train,
        traj_test,
        traj_true,
        flow_train,
        state_feature,
        action_feature,
        params,
        reward_params,
        reward_model=reward_model,
        gamma=gamma,
    )
    print("Final evaluation:", results)


if __name__ == "__main__":
    main()
