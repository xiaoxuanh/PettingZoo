import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical

from pettingzoo.investesg import investesg_v0

class BaseAgent(nn.Module):
    def __init__(self, observation_size, num_actions, hidden_sizes):
        super().__init__()
        layers = []
        input_size = observation_size
        for hidden_size in hidden_sizes:
            layers.append(self._layer_init(nn.Linear(input_size, hidden_size)))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.network = nn.Sequential(*layers)
        self.actor = self._layer_init(nn.Linear(hidden_sizes[-1], num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(hidden_sizes[-1], 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

class CompanyAgent(BaseAgent):
    def __init__(self, observation_size, num_actions):
        """For company agent num_actions is the number of discrete possible actions."""
        super().__init__(observation_size, num_actions, hidden_sizes=[256, 256, 128])

    def get_action_and_value(self, x, action=None):
        """For each action, returns logit."""
        x = x.float()
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

class InvestorAgent(BaseAgent):
    def __init__(self, observation_size, num_companies):
        """For investor agent num_actions is the size of the MultiDiscrete action space. 
        More specifically, it is the number of companies to invest in * Each action being binary."""
        num_actions = num_companies * 2
        super().__init__(observation_size, num_actions, hidden_sizes=[256, 256, 128])
        self.num_companies = num_companies

    def get_action_and_value(self, x, action=None):
        """For each company, returns logit likelihodd of investing."""
        x = x.float()
        hidden = self.network(x)
        logits = self.actor(hidden)  # Output logits directly
        # Reshape logits to (batch_size, num_companies, 2) for binary decision
        logits = logits.view(-1, self.num_companies, 2)

        # Create a list of Categorical distributions for each company
        # three dimensions: batch_size, num_companies, 2
        probs = [Categorical(logits=logits[:, i, :]) for i in range(logits.size(1))]
        if action is None:
            action = torch.stack([p.sample() for p in probs], dim=1)
        else:
            action = action.view(-1, self.num_companies)
        # logprobs and entropy are summed over the companies, computing the probability of the combined action
        logprobs = torch.stack([p.log_prob(a) for p, a in zip(probs, action.T)], dim=1)
        entropy = torch.stack([p.entropy() for p in probs], dim=1)
        
        return action, logprobs, entropy, self.critic(hidden)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)
    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    # Initialize a dictionary to hold the unbatched actions
    unbatched_actions = {}

    # Split the combined actions back into company and investor actions
    for i, agent in enumerate(env.possible_agents):
        if agent.startswith("company"):
            unbatched_actions[agent] = x[agent]
        elif agent.startswith("investor"):
            unbatched_actions[agent] = np.array(x[agent])

    return x


if __name__ == "__main__":
    
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    max_cycles = 125
    total_episodes = 2

    """ ENV SETUP """
    env = investesg_v0.InvestESG(num_companies = 3, num_investors = 3)
    # Separate company and investor agents
    company_agents = [agent for agent in env.possible_agents if agent.startswith("company")]
    investor_agents = [agent for agent in env.possible_agents if agent.startswith("investor")]

    # Ensure correct sizes
    assert len(company_agents) == env.num_companies
    assert len(investor_agents) == env.num_investors

    # Observation and action sizes
    company_observation_size = env.observation_space(company_agents[0]).shape[0]
    investor_observation_size = env.observation_space(investor_agents[0]).shape[0]

    company_actions = env.action_space(company_agents[0]).n # Assuming Discrete
    investor_actions_num_companies = len(env.action_space(investor_agents[0]).nvec)  # Assuming MultiDiscrete

    """ LEARNER SETUP """
    company_agent = CompanyAgent(observation_size=company_observation_size, num_actions=company_actions).to(device)
    investor_agent = InvestorAgent(observation_size=investor_observation_size, num_companies=investor_actions_num_companies).to(device)
    optimizer = optim.Adam(list(company_agent.parameters()) + list(investor_agent.parameters()), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return_company = torch.zeros(len(company_agents)).to(device)
    total_episodic_return_investor = torch.zeros(len(investor_agents)).to(device)
    """ COMPANY AGENT STORAGE """
    company_rb_obs = torch.zeros((max_cycles, len(company_agents), company_observation_size)).to(device)
    company_rb_actions = torch.zeros((max_cycles, len(company_agents))).to(device)
    company_rb_logprobs = torch.zeros((max_cycles, len(company_agents))).to(device)
    company_rb_rewards = torch.zeros((max_cycles, len(company_agents))).to(device)
    company_rb_terms = torch.zeros((max_cycles, len(company_agents))).to(device)
    company_rb_values = torch.zeros((max_cycles, len(company_agents))).to(device)
    """ INVESTOR AGENT STORAGE """
    investor_rb_obs = torch.zeros((max_cycles, len(investor_agents), investor_observation_size)).to(device)
    investor_rb_actions = torch.zeros((max_cycles, len(investor_agents), len(env.action_space(investor_agents[0]).nvec))).to(device)
    investor_rb_logprobs = torch.zeros((max_cycles, len(investor_agents), len(env.action_space(investor_agents[0]).nvec))).to(device)
    investor_rb_rewards = torch.zeros((max_cycles, len(investor_agents))).to(device)
    investor_rb_terms = torch.zeros((max_cycles, len(investor_agents))).to(device)
    investor_rb_values = torch.zeros((max_cycles, len(investor_agents))).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return_company = torch.zeros(len(company_agents)).to(device)
            total_episodic_return_investor = torch.zeros(len(investor_agents)).to(device)

            # each episode has num_steps
            for step in range(0, max_cycles):
                # Separate observations for companies and investors
                company_obs = {k: next_obs[k] for k in company_agents}
                investor_obs = {k: next_obs[k] for k in investor_agents}
                company_actions, company_logprobs, _, company_values = company_agent.get_action_and_value(
                batchify_obs(company_obs, device)
                )
                investor_actions, investor_logprobs, _, investor_values = investor_agent.get_action_and_value(
                batchify_obs(investor_obs, device)
                )

                # Combine actions
                combined_actions = {**{k: company_actions[i].item() for i, k in enumerate(company_agents)}, 
                                **{k: investor_actions[i].cpu().numpy() for i, k in enumerate(investor_agents)}}
                next_obs, rewards, terms, truncs, infos = env.step(unbatchify(combined_actions, env))

                # add to episode storage
                for i, agent in enumerate(company_agents):
                    company_rb_obs[step, i] = torch.tensor(company_obs[agent]).to(device)
                    company_rb_rewards[step, i] = rewards[agent]
                    company_rb_terms[step, i] = terms[agent]
                    company_rb_actions[step, i] = company_actions[i]
                    company_rb_logprobs[step, i] = company_logprobs[i]
                    company_rb_values[step, i] = company_values[i]

                for i, agent in enumerate(investor_agents):
                    investor_rb_obs[step, i] = torch.tensor(investor_obs[agent]).to(device)
                    investor_rb_rewards[step, i] = rewards[agent]
                    investor_rb_terms[step, i] = terms[agent]
                    investor_rb_actions[step, i] = torch.tensor(investor_actions[i].cpu().numpy()).to(device)
                    investor_rb_logprobs[step, i] = torch.tensor(investor_logprobs[i].cpu().numpy()).to(device)
                    investor_rb_values[step, i] = investor_values[i]

                # compute episodic return
                total_episodic_return_company += sum([rewards[agent] for agent in company_agents])
                total_episodic_return_investor += sum([rewards[agent] for agent in investor_agents])

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            company_rb_advantages = torch.zeros_like(company_rb_rewards).to(device)
            investor_rb_advantages = torch.zeros_like(investor_rb_rewards).to(device)

            for t in reversed(range(end_step)):
                delta_company = (
                    company_rb_rewards[t]
                    + gamma * company_rb_values[t + 1] * company_rb_terms[t + 1]
                    - company_rb_values[t]
                )
                company_rb_advantages[t] = delta_company + gamma * gamma * company_rb_advantages[t + 1]

                delta_investor = (
                    investor_rb_rewards[t]
                    + gamma * investor_rb_values[t + 1] * investor_rb_terms[t + 1]
                    - investor_rb_values[t]
                )
                investor_rb_advantages[t] = delta_investor + gamma * gamma * investor_rb_advantages[t + 1]

            company_rb_returns = company_rb_advantages + company_rb_values
            investor_rb_returns = investor_rb_advantages + investor_rb_values

        # convert our episodes to batch of individual transitions
        b_company_obs = torch.flatten(company_rb_obs[:end_step], start_dim=0, end_dim=1)
        b_company_logprobs = torch.flatten(company_rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_company_actions = torch.flatten(company_rb_actions[:end_step], start_dim=0, end_dim=1)
        b_company_returns = torch.flatten(company_rb_returns[:end_step], start_dim=0, end_dim=1)
        b_company_values = torch.flatten(company_rb_values[:end_step], start_dim=0, end_dim=1)
        b_company_advantages = torch.flatten(company_rb_advantages[:end_step], start_dim=0, end_dim=1)

        b_investor_obs = torch.flatten(investor_rb_obs[:end_step], start_dim=0, end_dim=1)
        b_investor_logprobs = torch.flatten(investor_rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_investor_actions = torch.flatten(investor_rb_actions[:end_step], start_dim=0, end_dim=1)
        b_investor_returns = torch.flatten(investor_rb_returns[:end_step], start_dim=0, end_dim=1)
        b_investor_values = torch.flatten(investor_rb_values[:end_step], start_dim=0, end_dim=1)
        b_investor_advantages = torch.flatten(investor_rb_advantages[:end_step], start_dim=0, end_dim=1)

        b_index_company = np.arange(len(b_company_obs))
        b_index_investor = np.arange(len(b_investor_obs))

        clip_fracs = []
        for repeat in range(3):
            np.random.shuffle(b_index_company)
            np.random.shuffle(b_index_investor)

            for start in range(0, len(b_company_obs), batch_size):
                end = start + batch_size
                batch_index_company = b_index_company[start:end]

                _, newlogprob, entropy, value = company_agent.get_action_and_value(
                    b_company_obs[batch_index_company], b_company_actions.long()[batch_index_company]
                )
                logratio = newlogprob - b_company_logprobs[batch_index_company]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                advantages = b_company_advantages[batch_index_company]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_company_returns[batch_index_company]) ** 2
                v_clipped = b_company_values[batch_index_company] + torch.clamp(
                    value - b_company_values[batch_index_company],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_company_returns[batch_index_company]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for start in range(0, len(b_investor_obs), batch_size):
                end = start + batch_size
                batch_index_investor = b_index_investor[start:end]

                _, newlogprob, entropy, value = investor_agent.get_action_and_value(
                    b_investor_obs[batch_index_investor], b_investor_actions.long()[batch_index_investor]
                )
                logratio = newlogprob - b_investor_logprobs[batch_index_investor]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                advantages = b_investor_advantages[batch_index_investor]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value = value.flatten()
                v_loss_unclipped = (value - b_investor_returns[batch_index_investor]) ** 2
                v_clipped = b_investor_values[batch_index_investor] + torch.clamp(
                    value - b_investor_values[batch_index_investor],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_investor_returns[batch_index_investor]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred_company, y_true_company = b_company_values.cpu().numpy(), b_company_returns.cpu().numpy()
        var_y_company = np.var(y_true_company)
        explained_var_company = np.nan if var_y_company == 0 else 1 - np.var(y_true_company - y_pred_company) / var_y_company

        y_pred_investor, y_true_investor = b_investor_values.cpu().numpy(), b_investor_returns.cpu().numpy()
        var_y_investor = np.var(y_true_investor)
        explained_var_investor = np.nan if var_y_investor == 0 else 1 - np.var(y_true_investor - y_pred_investor) / var_y_investor

        print(f"Training episode {episode}")
        print(f"Company Episodic Return: {total_episodic_return_company.sum().item()}")
        print(f"Investor Episodic Return: {total_episodic_return_investor.sum().item()}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Company Value Loss: {v_loss.item()}")
        print(f"Company Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance Company: {explained_var_company.item()}")
        print(f"Explained Variance Investor: {explained_var_investor.item()}")
        print("\n-------------------------------------------\n")

    """ RENDER THE POLICY """
    env = investesg_v0.InvestESG(num_companies=3, num_investors=3)

    # Separate company and investor agents
    company_agents = [agent for agent in env.possible_agents if agent.startswith("company")]
    investor_agents = [agent for agent in env.possible_agents if agent.startswith("investor")]

    company_agent.eval()
    investor_agent.eval()

    with torch.no_grad():
        # Render 5 episodes
        for episode in range(5):
            obs, infos = env.reset(seed=None)
            obs = {agent: torch.tensor(obs[agent]).to(device) for agent in env.possible_agents}

            terms = [False]
            truncs = [False]
            
            while not any(terms) and not any(truncs):
                # Separate observations for companies and investors
                company_obs = {k: obs[k] for k in company_agents}
                investor_obs = {k: obs[k] for k in investor_agents}

                company_actions, _, _, _ = company_agent.get_action_and_value(
                    batchify_obs(company_obs, device)
                )
                investor_actions, _, _, _ = investor_agent.get_action_and_value(
                    batchify_obs(investor_obs, device)
                )

                # Combine actions
                combined_actions = {**{k: company_actions[i].item() for i, k in enumerate(company_agents)}, 
                                    **{k: investor_actions[i].cpu().numpy() for i, k in enumerate(investor_agents)}}

                obs, rewards, terms, truncs, infos = env.step(unbatchify(combined_actions, env))
                obs = {agent: torch.tensor(obs[agent]).to(device) for agent in env.possible_agents}

                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]

                # Optional: Print or log information here for debugging or analysis
                print(f"Episode {episode}, Steps {len(terms)}")
