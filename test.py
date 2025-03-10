import torch 
from acornrl.models.huggingface import HFModel
from acornrl.agents.actor_agent import ActorAgent
from acornrl.collectors import SequentialTextArenaCollector
from acornrl.trainers.standard_trainer import StandardTrainer

MODEL_NAME = "roneneldan/TinyStories-1M" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EPOCHS = 3
EPISODES_PER_EPOCH = 3

# initialize the LLM
model = HFModel(model_name=MODEL_NAME, device="cuda")

# test generator
# print(model.generate_text(prompt="Tell my about love: "))


# initialize the optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-10)


# # wrap the llm in the ActorAgent
agent = ActorAgent(model=model)

# test agent
# print(agent("The secret of love is all about: "))

# # initialize the collector
collector = SequentialTextArenaCollector(env_ids=["TicTacToe-v0"])

# test data collection
# data = collector.collect(agent1=agent, agent2=agent, num_episodes=2) # dict_keys(['player_id', 'observation', 'reasoning', 'action', 'step', 'final_reward'])

# create the trainer 
trainer = StandardTrainer(
    agent=agent,
    optimizer=optimizer,
    collector=collector
)

# start training
trainer.train(
    epochs=EPOCHS,
    episodes_per_epoch=EPISODES_PER_EPOCH
)