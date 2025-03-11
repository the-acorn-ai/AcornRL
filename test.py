import torch 
from acornrl.models.huggingface import HFModel
from acornrl.agents.actor_agent import ActorAgent
from acornrl.collectors import SequentialTextArenaCollector
from acornrl.trainers.standard_trainer import StandardTrainer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EPOCHS = 3
EPISODES_PER_EPOCH = 3

MAX_LENGTH = 8192


# initialize the LLM
model = HFModel(
    model_name=MODEL_NAME, 
    device="cuda", 
    max_length=MAX_LENGTH,
    use_qlora=True,
)

# test generator
# print(model.generate_text(prompt="Tell my about love: "))


# initialize the optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-10)


# # wrap the llm in the ActorAgent
agent = ActorAgent(model=model, max_new_tokens=MAX_LENGTH)

# test agent
# print(agent("The secret of love is all about: "))

# # initialize the collector
collector = SequentialTextArenaCollector(env_ids=["SpellingBee-v0"])

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