import numpy as np







def reshape_rewards(data_list, transformations=None, normalize=False):
    # add a reward step count
    # for idx in range(len(data_list)):
    #     # reward more turns
    #     data_list[idx]["final_reward"] = data_list[idx]["final_reward"] + data_list[idx]["full_length"]*0.05

    #     # punish for not returning reasoning
    #     if data_list[idx]["reasoning"] == "":
    #         data_list[idx]["final_reward"] = data_list[idx]["final_reward"] - 1


    if normalize:
        # lastly normalize the reward/advantage
        rewards = [entry["final_reward"] for entry in data_list]

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        normalize = lambda x: (x-avg_reward) / (std_reward+1e-8)
        for idx in range(len(data_list)):
            data_list[idx]["final_reward"] = normalize(data_list[idx]["final_reward"])

    return data_list









