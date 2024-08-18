from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, MultiDiscrete, Box
import functools

import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns
import itertools

class Company:
    def __init__(self, capital=10000, climate_risk_exposure = 0.5, beta = 0.1667, cont_esg_score=False):
        self.initial_capital = capital      # initial capital
        self.capital = capital              # current capital
        self.beta = beta                    # Beta risk factor against market performance
        self.cont_esg_score = cont_esg_score # Continous ESG score

        self.initial_climate_risk_exposure \
            = climate_risk_exposure         # initial climate risk exposure
        self.climate_risk_exposure \
            = climate_risk_exposure         # capital loss ratio when a climate event occurs
        
        self.exposure_decay_rate = 0.00001      # decay rate of climate risk exposure
        self.esg_invested = 0               # cumulative amount invested in ESG
        self.margin = 0                     # single period profit margin
        self.capital_gain = 0               # single period capital gain
        self.strategy = None                # 1: "mitigation", 2: "greenwashing", 0: "none"
        self.esg_score = 0               # signal to be broadcasted to investors: "esg-friendly":1, "none":0

    def receive_investment(self, amount):
        """Receive investment from investors."""
        self.capital += amount

    def lose_investment(self, amount):
        """Lose investment due to climate event."""
        self.capital -= amount
    
    def make_decision(self, strategy):
        """Make a decision on how to allocate capital."""
        self.strategy = strategy
        if strategy == 1:
            self.invest_in_esg(self.capital*0.025)  
            # TODO: this is a hardcoded value, should be a parameter;
            # also if hardcode is to be changed, should change in update_capital function as well
        elif strategy == 2:
            self.invest_in_greenwash(self.capital*0.01)  
            # TODO: this is a hardcoded value, should be a parameter
            # also if hardcode is to be changed, should change in update_capital function as well
        else:
            self.esg_score = 0
        
        if self.cont_esg_score:
            self.update_esg_score()
    
    def update_esg_score(self):
        self.esg_score = self.esg_invested / self.capital
  
    def invest_in_esg(self, amount):
        """Invest a certain amount in ESG."""
        self.esg_invested += amount
        self.capital -= amount
        # climate risk exposure is an exponential decay function of the amount invested in ESG
        self.climate_risk_exposure = self.initial_climate_risk_exposure \
            * np.exp(-self.exposure_decay_rate * self.esg_invested)
        self.esg_score = 1

    def invest_in_greenwash(self, amount):
        """Invest a certain amount in greenwashing."""
        self.capital -= amount
        self.esg_score = 1

    def update_capital(self, environment):
        """Update the capital based on market performance and climate event."""
        # add a random disturbance to market performance
        company_performance = np.random.normal(loc=environment.market_performance, scale=self.beta) 
        # ranges from 0.5 to 1.5 of market performance baseline most of time
        new_capital = self.capital * company_performance
        if environment.climate_event_occurrence > 0:
            new_capital *= (1 - self.climate_risk_exposure)**environment.climate_event_occurrence

        # backout the original capital based on esg investment
        if self.strategy == 1:
            base_capital = self.capital / 0.975 
            # TODO: this is a hardcoded value, should be a parameter; 
            # also if hardcode is to be changed, should change in make_decision function as well
        elif self.strategy == 2:
            base_capital = self.capital / 0.99
        else:
            base_capital = self.capital
        
        # calculate margin and capital gain
        self.capital_gain = new_capital - base_capital # ending capital - starting capital
        self.margin = self.capital_gain/base_capital
        self.capital = new_capital
        
    
    def reset(self):
        """Reset the company to the initial state."""
        self.capital = self.initial_capital
        self.climate_risk_exposure = self.initial_climate_risk_exposure
        self.esg_invested = 0
        self.margin = 0
        self.capital_gain = 0
        self.strategy = None
        self.esg_score = 0
    
class Investor:
    def __init__(self, capital=10, esg_preference=0.5):
        self.initial_capital = capital      # initial capital
        self.capital = capital              # current capital
        self.investments = {}               # dictionary to track investments in different companies
        self.esg_preference = esg_preference # the weight of ESG in the investor's decision making
        self.utility = 0                     # single-period reward
    
    def initial_investment(self, environment):
        """Invest in all companies at the beginning of the simulation."""
        self.investments = {i: 0 for i in range(environment.num_companies)}
    
    def invest(self, amount, company_idx):
        """Invest a certain amount in a company. 
        At the end of each period, investors collect all returns and then redistribute capital in next round."""
        if self.capital+1e-6 < amount:
            raise ValueError("Investment amount exceeds available capital.")
        else:
            self.capital -= amount
            self.investments[company_idx] += amount

    def divest(self, company_idx, environment):
        """Divest from a company."""
        investment_return = self.investments[company_idx]
        self.capital += investment_return
        environment.companies[company_idx].lose_investment(investment_return)
        self.investments[company_idx] = 0
    
    def calculate_utility(self, environment):
        """Calculate reward based on market performance and ESG preferences."""
        returns = 0
        esg_reward = 0
        base_capital = sum(self.investments.values())
        if base_capital == 0:
            self.utility = 0
        else:
            for company_idx, investment in self.investments.items():
                if investment == 0:
                    continue
                company = environment.companies[company_idx]
                returns += investment * company.margin
                esg_reward += company.esg_score
                # update value of investment based on returns; the worst case is to lose all investment
                self.investments[company_idx] = max(investment * (1 + company.margin), 0)

            overall_return_rate = returns/base_capital
            utility = overall_return_rate + self.esg_preference * esg_reward
            self.utility = utility

    
    def reset(self):
        """Reset the investor to the initial state."""
        self.capital = self.initial_capital
        self.investments = {i: 0 for i in self.investments}
        self.utility = 0


class InvestESG(ParallelEnv):
    """
    ESG investment environment.
    """

    metadata = {"name": "InvestESG"}

    def __init__(
        self,
        company_attributes=None,
        investor_attributes=None,
        num_companies=10,
        num_investors=10,
        initial_climate_event_probability=0.1,
        max_steps=100,
        market_performance_baseline=1.1, 
        market_performance_variance=0.0
    ):
        self.max_steps = max_steps
        self.timestamp = 0

        # initialize companies and investors based on attributes if not None
        if company_attributes is not None:
            self.companies = [Company(**attributes) for attributes in company_attributes]
            self.num_companies = len(company_attributes)
        else:
            self.companies = [Company() for _ in range(num_companies)]
            self.num_companies = num_companies
        
        if investor_attributes is not None:
            self.investors = [Investor(**attributes) for attributes in investor_attributes]
            self.num_investors = len(investor_attributes)
        else:
            self.num_investors = num_investors
            self.investors = [Investor() for _ in range(num_investors)]
        
        self.agents = [f"company_{i}" for i in range(self.num_companies)] + [f"investor_{i}" for i in range(self.num_investors)]
        self.n_agents = len(self.agents)
        self.possible_agents = self.agents[:]
        self.market_performance_baseline = market_performance_baseline # initial market performance
        self.market_performance_variance = market_performance_variance # variance of market performance
        self.initial_climate_event_probability = initial_climate_event_probability # initial probability of climate event
        self.climate_event_probability = initial_climate_event_probability # current probability of climate event
        self.climate_event_occurrence = 0 # number of climate events occurred in the current step
        # initialize investors with initial investments dictionary
        for investor in self.investors:
            investor.initial_investment(self)

        # initialize historical data storage
        self.history = {
            "esg_investment": [],
            "climate_risk": [],
            "climate_event_occurs": [],
            "market_performance": [],
            "market_total_wealth": [],
            "company_capitals": [[] for _ in range(self.num_companies)],
            "company_climate_risk": [[] for _ in range(self.num_companies)],
            "investor_capitals": [[] for _ in range(self.num_investors)],
            "investor_utility": [[] for _ in range(self.num_investors)],
            "investment_matrix": np.zeros((self.num_investors, self.num_companies)),
            "company_decisions": [[] for _ in range(self.num_companies)]
        }


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # each company has 3 possible actions: "mitigation", "greenwashing", "none"
        # each investor has num_companies possible*2 actions: for each company, invest/not invest
        # if agent is a company
        if agent.startswith("company"):
            return Discrete(3)
        # if agent is an investor
        else:
            return MultiDiscrete(self.num_companies*[2])
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # all agents have access to the same information, namely the capital, climate risk exposure, ESG score, and margin of each company
        # of all companies and the investment in each company and remaining capital of each investor
        observation_size = self.num_companies * 4 + self.num_investors * (self.num_companies + 1)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(observation_size,))
        return observation_space

    def step(self, actions):
        """Step function for the environment."""

        ## Temporary Code: TO BE REMOVED LATER
        rng1 = np.random.default_rng(self.timestamp)
        rng2 = np.random.default_rng(self.timestamp*1000)

        ## unpack actions
        # first num_companies actions are for companies, the rest are for investors
        companys_actions = dict(itertools.islice(actions.items(), self.num_companies))
        remaining_actions = {k: v for k, v in actions.items() if k not in companys_actions}
        # Reindex investor actions to start from 0
        investors_actions = {f"investor_{i}": action for i, (k, action) in enumerate(remaining_actions.items())}

        ## action masks
        # if company has negative capital, it cannot invest in ESG or greenwashing
        for i, company in enumerate(self.companies):
            if company.capital < 0:
                companys_actions[f"company_{i}"] = 0

        # 0. investors divest from all companies and recollect capital
        for investor in self.investors:
            for company in investor.investments:
                investor.divest(company, self)

        # 1. investors allocate capital to companies (binary decision to invest/not invest)
        for i, investor in enumerate(self.investors):
            investor_action = investors_actions[f"investor_{i}"]
            # number of companies that the investor invests in
            num_investments = np.sum(investor_action)
            if num_investments > 0:
                # equal investment in each company; round down to nearest integer to avoid exceeding capital
                investment_amount = np.floor(investor.capital/num_investments) 
                for j, company in enumerate(self.companies):
                    if investor_action[j]:
                        investor.invest(investment_amount, j)
                        # company receives investment
                        company.receive_investment(investment_amount)
                   
        # 2. companies invest in ESG/greenwashing/none, report margin and esg score
        for i, company in enumerate(self.companies):
            company_action = companys_actions[f"company_{i}"]
            company.make_decision(company_action)

        # 3. update probabilities of climate event based on cumulative ESG investments across companies
        total_esg_investment = np.sum(np.array([company.esg_invested for company in self.companies]))
        self.climate_event_probability =  self.initial_climate_event_probability + 0.014*self.timestamp/(1+0.028*total_esg_investment)

        # 4. market performance and climate event evolution
        self.market_performance = rng1.normal(loc=self.market_performance_baseline, scale=self.market_performance_variance)   # ranges from 0.9 to 1.1 most of time
        # TODO: consider other distributions and time-correlation of market performance
        self.climate_event_occurrence = int(self.climate_event_probability) + (rng2.random() < self.climate_event_probability % 1).astype(int)

        # 5. companies update capital based on market performance and climate event
        for company in self.companies:
            company.update_capital(self)

        # 6. investors calculate returns based on market performance
        for investor in self.investors:
            investor.calculate_utility(self)

        # 7. termination and truncation
        self.timestamp += 1
        termination = {agent: self.timestamp >= self.max_steps for agent in self.agents}
        truncation = termination

        observations = self._get_observation()
        rewards = self._get_reward()
        infos = self._get_infos()

        # 8. update history
        self._update_history()
        
        if any(termination.values()):
            self.agents = []
        
        # 8. update observation for each company and investor
        return observations, rewards, termination, truncation, infos

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        for company in self.companies:
            company.reset()
        for investor in self.investors:
            investor.reset()
        self.agents = [f"company_{i}" for i in range(self.num_companies)] + [f"investor_{i}" for i in range(self.num_investors)]
        self.market_performance = 1
        self.climate_event_probability = self.initial_climate_event_probability
        self.climate_event_occurrence = 0
        self.timestamp = 0
        # reset historical data
        self.history = {
            "esg_investment": [],
            "climate_risk": [],
            "climate_event_occurs": [],
            "market_performance": [],
            "market_total_wealth": [], 
            "company_capitals": [[] for _ in range(self.num_companies)],
            "company_climate_risk": [[] for _ in range(self.num_companies)],
            "investor_capitals": [[] for _ in range(self.num_investors)],
            "investor_utility": [[] for _ in range(self.num_investors)],
            "investment_matrix": np.zeros((self.num_investors, self.num_companies)),
            "company_decisions": [[] for _ in range(self.num_companies)]
        }
        self.fig = None
        self.ax = None

        return self._get_observation(), self._get_infos()
    
    def _get_observation(self):
        """Get observation for each company and investor. Public information is shared across all agents."""
        # Collect company observations
        company_obs = []
        for company in self.companies:
            company_obs.extend([company.capital, company.climate_risk_exposure, company.esg_score, company.margin])
        # Collect investor observations
        investor_obs = []
        for investor in self.investors:
            investor_obs.extend(list(investor.investments.values()) + [investor.capital])
        full_obs = np.array(company_obs + investor_obs)

        # Return the same observation for all agents
        return {agent: full_obs for agent in self.agents}

    def _get_reward(self):
        """Get reward for all agents."""
        rewards = {}
        for i, company in enumerate(self.companies):
            rewards[f"company_{i}"] = company.capital_gain #TODO: ideally, we should remove investor principals from company capitals
        for i, investor in enumerate(self.investors):
            rewards[f"investor_{i}"] = investor.utility
        return rewards
    
    def _get_infos(self):
        """Get infos for all agents. Dummy infos for compatibility with pettingzoo."""
        infos = {agent: {} for agent in self.agents}
        return infos

    def _update_history(self):
        """Update historical data."""
        self.history["esg_investment"].append(sum(company.esg_invested for company in self.companies))
        self.history["climate_risk"].append(self.climate_event_probability)
        self.history["climate_event_occurs"].append(self.climate_event_occurrence)
        self.history["market_performance"].append(self.market_performance)
        self.history["market_total_wealth"].append(sum(company.capital for company in self.companies)+sum(investor.capital for investor in self.investors))
        for i, company in enumerate(self.companies):
            self.history["company_capitals"][i].append(company.capital)
            self.history["company_decisions"][i].append(company.strategy)
            self.history["company_climate_risk"][i].append(company.climate_risk_exposure)
        for i, investor in enumerate(self.investors):
            self.history["investor_capitals"][i].append(investor.capital+sum(investor.investments.values()))
            self.history["investor_utility"][i].append(investor.utility)
            for j, investment in investor.investments.items():
                self.history["investment_matrix"][i, j] += investment

    @property
    def name(self) -> str:
        """Environment name."""
        return "InvestESG"

    def render(self, mode='human', fig='fig'):
        # import pdb; pdb.set_trace()
        
        if not hasattr(self, 'fig') or self.fig is None:
            # Initialize the plot only once
            self.fig = Figure(figsize=(36, 10))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.subplots(2, 5)  # Adjusted to 2 rows and 4 columns
            plt.subplots_adjust(hspace=0.5, wspace=1)  # Adjust spacing as needed
            plt.ion()  # Turn on interactive mode for plotting

        # Ensure self.ax is always a list of axes
        if not isinstance(self.ax, np.ndarray):
            self.ax = np.array([self.ax])

        # Clear previous figures to update with new data
        for row in self.ax:
            for axis in row:
                axis.cla()

        # Subplot 1: Overall ESG Investment and Climate Risk over time
        ax1 = self.ax[0][0]
        ax2 = ax1.twinx()  # Create a secondary y-axis

        ax1.plot(self.history["esg_investment"], label='Cumulative ESG Investment', color='blue')
        ax2.plot(self.history["climate_risk"], label='Climate Risk', color='orange')
        # Add vertical lines for climate events
        for i, event in enumerate(self.history["climate_event_occurs"]):
            if event==1:
                ax1.axvline(x=i, color='orange', linestyle='--', alpha=0.5)
            if event>1:
                ax1.axvline(x=i, color='red', linestyle='--', alpha=0.5)

        ax1.set_title('Overall Metrics Over Time')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Investment in ESG')
        ax2.set_ylabel('Climate Event Probability')
        ax2.set_ylim(0, 2)  # Set limits for Climate Event Probability

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Subplot 2: Company Decisions
        for i, decision_history in enumerate(self.history["company_decisions"]):
            self.ax[0][1].plot(decision_history, label=f'Company {i}', linestyle='None', marker='o')
        self.ax[0][1].set_title('Company Decisions Over Time')
        self.ax[0][1].set_ylabel('Decision')
        self.ax[0][1].set_xlabel('Timestep')
        self.ax[0][1].set_yticks([0, 1, 2])
        self.ax[0][1].set_yticklabels(['None', 'Mitigation', 'Greenwashing'], rotation=90)
        self.ax[0][1].legend(loc='upper right')

        # Subplot 3: Company Climate risk exposure over time
        for i, climate_risk_history in enumerate(self.history["company_climate_risk"]):
            self.ax[0][2].plot(climate_risk_history, label=f'Company {i}')
        self.ax[0][2].set_title('Company Climate Risk Over Time')
        self.ax[0][2].set_ylabel('Climate Risk')
        self.ax[0][2].set_xlabel('Timestep')
        self.ax[0][2].legend(loc='upper right')

        # Subplot 4: Company Capitals over time
        for i, capital_history in enumerate(self.history["company_capitals"]):
            self.ax[0][3].plot(capital_history, label=f'Company {i}')
        self.ax[0][3].set_title('Company Capitals Over Time')
        self.ax[0][3].set_ylabel('Capital')
        self.ax[0][3].set_xlabel('Timestep')
        self.ax[0][3].legend(loc='upper right')

        # Subplot 5: Investment Matrix
        investment_matrix = self.history["investment_matrix"]
        ax = self.ax[1][0]
        sns.heatmap(investment_matrix, ax=ax, cmap='Reds', cbar=True, annot=True, fmt='g')

        ax.set_title('Investment Matrix')
        ax.set_ylabel('Investor ID')
        ax.set_xlabel('Company ID')

         # Subplot 6: Investor Capitals over time
        for i, capital_history in enumerate(self.history["investor_capitals"]):
            self.ax[1][1].plot(capital_history, label=f'Investor {i}')
        self.ax[1][1].set_title('Investor Capitals Over Time')
        self.ax[1][1].set_ylabel('Capital')
        self.ax[1][1].set_xlabel('Timestep')
        self.ax[1][1].legend(loc='upper right')

        # Subplot 7: Investor Utility over time
        for i, utility_history in enumerate(self.history["investor_utility"]):
            self.ax[1][2].plot(utility_history, label=f'Investor {i}')
        self.ax[1][2].set_title('Investor Utility Over Time')
        self.ax[1][2].set_ylabel('Utility')
        self.ax[1][2].set_xlabel('Timestep')
        self.ax[1][2].legend(loc='upper right')

        # Subplot 7: Cumulative Investor Utility over time
        for i, utility_history in enumerate(self.history["investor_utility"]):
            cumulative_utility_history = list(itertools.accumulate(utility_history))
            self.ax[1][3].plot(cumulative_utility_history, label=f'Investor {i}')
        self.ax[1][3].set_title('Cumulative Investor Utility Over Time')
        self.ax[1][3].set_ylabel('Cumulative Utility')
        self.ax[1][3].set_xlabel('Timestep')
        self.ax[1][3].legend(loc='upper right')

        # Subplot 8: Market Total Wealth over time
        self.ax[1][4].plot(self.history["market_total_wealth"], label='Total Wealth', color='green')
        self.ax[1][4].set_title('Market Total Wealth Over Time')
        self.ax[1][4].set_ylabel('Total Wealth')
        self.ax[1][4].set_xlabel('Timestep')
        self.ax[1][4].legend()


        # Update the plots
        self.canvas.draw()
        self.canvas.flush_events()
        plt.pause(0.001)  # Pause briefly to update plots

        # TODO: Consider generate videos later
        if mode == 'human':
            plt.show(block=False)
        elif mode == 'rgb_array':
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            images = wandb.Image(img)
            wandb.log({"figure": images})
            d = {
                "total climate_event_occurs": sum(self.history["climate_event_occurs"]),
                "final climate risk": self.history["climate_risk"][-1],
                "cumulative climate risk": sum(self.history["climate_risk"]),
                "final esg_investment": self.history['esg_investment'][-1]
            }
            for i, company_climate_risk_history in enumerate(self.history['company_climate_risk']):
                d[f"final company_{i} climate risk"] = company_climate_risk_history[-1]
            
            for i, investor_utility_history in enumerate(self.history['investor_utility']):
                d[f"cumulative investor_{i} utility"] = sum(investor_utility_history)
            wandb.log(d)
            return img
        
        

if __name__ == "__main__":
    env = InvestESG(company_attributes=[{'capital':10000,'climate_risk_exposure':0.5,'beta':0},
                                    {'capital':10000,'climate_risk_exposure':0.5,'beta':0},
                                    {'capital':10000,'climate_risk_exposure':0.5,'beta':0}], 
                                    num_investors=3, initial_climate_event_probability=0.1,
                                    market_performance_baseline=1.05, market_performance_variance=0)
    env.reset()
    company_actions = {f"company_{i}": env.action_space(f"company_{i}").sample() for i in range(env.num_companies)}
    # company 0 never does anything
    company_actions['company_0'] = 0
    company_actions['company_1'] = 0
    company_actions['company_2'] = 0
    investor_actions = {f"investor_{i}": env.action_space(f"investor_{i}").sample() for i in range(env.num_investors)}
    # mask such that investor 0 only invests in company 0
    investor_actions['investor_0'] = [1, 0, 0]
    investor_actions['investor_1'] = [0, 1, 0]
    investor_actions['investor_2'] = [0, 0, 1]
    actions = {**company_actions, **investor_actions}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    for _ in range(100):
        obs, rewards, terminations, truncations, infos = env.step(actions)
    img = env.render(mode='rgb_array')
    import pdb; pdb.set_trace()

