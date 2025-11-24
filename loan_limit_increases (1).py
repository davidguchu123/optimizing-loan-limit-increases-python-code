state_cols = [c for c in df.columns if any(k in c.lower() for k in ['state','risk','bucket','score_bucket','credit_state'])]
markov_matrix = None
if len(state_cols) >= 2:
    s_cur, s_next = state_cols[0], state_cols[1]
    if s_cur in df.columns and s_next in df.columns:
        cm_tab = pd.crosstab(df[s_cur], df[s_next], normalize='index')
        markov_matrix = cm_tab
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
INPUT_FILE =r"C:\Users\Administrator\Downloads\loan_limit_increases.xlsx"
RANDOM_STATE = 42
df=pd.read_excel(INPUT_FILE)
print(df.isna().sum().sort_values(ascending=False).head(20))
average_percentage = df['On-time Payments (%)'].mean()
print(f"Calculated Average On-time Payments: {average_percentage:.2f}%\n")
threshold_good = average_percentage + 5
threshold_bad = average_percentage - 5
conditions = [
    df['On-time Payments (%)'] >= threshold_good,
    (df['On-time Payments (%)'] < threshold_good) & (df['On-time Payments (%)'] > threshold_bad)
]
choices = [
    'good',    
    'average'   
]

df['probability distribution'] = np.select(
    condlist=conditions, 
    choicelist=choices,  
    default='bad'        
)

df.head()
le=LabelEncoder()
df['probability distribution']=le.fit_transform(df['probability distribution'])
df.head()
y = df['probability distribution'].values.reshape(-1, 1)
X = df.drop(['probability distribution'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
pipe_log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
pipe_rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
pipe_log.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)
y_pred_log = pipe_log.predict(X_test)
y_pred_rf = pipe_rf.predict(X_test)

acc_log = accuracy_score(y_test, y_pred_log)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Logistic R2-like accuracy: {acc_log:.3f}, RandomForest accuracy: {acc_rf:.3f}")
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LoanLimitOptimizer:
    def __init__(self, df, annual_discount_rate=0.19):
        self.data = df
        self.discount_rate = annual_discount_rate
        self.risk_states = ['Prime', 'Near-Prime', 'Subprime']
        self.setup_parameters()
        
    def setup_parameters(self):
        """Initialize model parameters"""
        # Transition probabilities (estimated from industry data)
        self.transition_probs = {
            'Prime': {'Prime': 0.85, 'Near-Prime': 0.12, 'Subprime': 0.03},
            'Near-Prime': {'Prime': 0.15, 'Near-Prime': 0.70, 'Subprime': 0.15},
            'Subprime': {'Prime': 0.05, 'Near-Prime': 0.25, 'Subprime': 0.70}
        }
        
        # Default probabilities by risk state
        self.default_probs = {'Prime': 0.01, 'Near-Prime': 0.05, 'Subprime': 0.15}
        
        # Profit parameters
        self.increase_profit = 40
        self.default_cost_multiplier = 3  
        self.max_increases_per_year = 6
        self.capital_requirement_ratio = 0.08
        
    def classify_risk_state(self, on_time_payment_rate):
        """Classify customers into risk states"""
        if on_time_payment_rate >= 95:
            return 'Prime'
        elif on_time_payment_rate >= 85:
            return 'Near-Prime'
        else:
            return 'Subprime'
    
    def calculate_expected_profit(self, loan_amount, risk_state, action):
        """Calculate expected profit for granting/denying increase"""
        if action == 0:  # Deny increase
            return 0
        
        default_prob = self.default_probs[risk_state]
        expected_profit = self.increase_profit * (1 - default_prob)
        expected_loss = loan_amount * self.default_cost_multiplier * default_prob
        
        return expected_profit - expected_loss
    
    def monte_carlo_simulation(self, n_simulations=1000, time_horizon=12):
        """Run Monte Carlo simulation for loan lifecycle"""
        results = []
        
        for _ in range(n_simulations):
             total_profit = 0
            current_capital = 1000000  # Initial capital
            increases_granted = 0
            
            for month in range(time_horizon):
                # Sample customers for this period
                sample_customers = self.data.sample(min(1000, len(self.data)))
                
                for _, customer in sample_customers.iterrows():
                    risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
                    
                    # Check eligibility (60 days since last loan)
                    if customer['Days Since Last Loan'] >= 60 and increases_granted < self.max_increases_per_year:
                        
                        # Calculate expected profit
                        expected_profit = self.calculate_expected_profit(
                            customer['Initial Loan ($)'], risk_state, 1
                        )
                        
                        # Grant increase if profitable and capital available
                        capital_required = customer['Initial Loan ($)'] * self.capital_requirement_ratio
                        
                        if expected_profit > 0 and current_capital >= capital_required:
                            total_profit += expected_profit
                            current_capital -= capital_required
                            increases_granted += 1
                            current_state = risk_state
                            next_state_probs = self.transition_probs[current_state]
                            next_state = np.random.choice(
                                list(next_state_probs.keys()),
                                p=list(next_state_probs.values())
                            )
            
            # Discount future profits
            discounted_profit = total_profit / ((1 + self.discount_rate/12) ** time_horizon)
            results.append(discounted_profit)
        
        return np.array(results)
                            
                            # Simulate state transition
            # Initialize simulation paramet# Cost
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LoanLimitOptimizer:
    def __init__(self, df, annual_discount_rate=0.19):
        self.data = df
        self.discount_rate = annual_discount_rate
        self.risk_states = ['Prime', 'Near-Prime', 'Subprime']
        self.setup_parameters()
        
    def setup_parameters(self):
        """Initialize model parameters"""
        # Transition probabilities (estimated from industry data)
        self.transition_probs = {
            'Prime': {'Prime': 0.85, 'Near-Prime': 0.12, 'Subprime': 0.03},
            'Near-Prime': {'Prime': 0.15, 'Near-Prime': 0.70, 'Subprime': 0.15},
            'Subprime': {'Prime': 0.05, 'Near-Prime': 0.25, 'Subprime': 0.70}
        }
        
        # Default probabilities by risk state
        self.default_probs = {'Prime': 0.01, 'Near-Prime': 0.05, 'Subprime': 0.15}
        
        # Profit parameters
        self.increase_profit = 40
        self.default_cost_multiplier = 3  
        self.max_increases_per_year = 6
        self.capital_requirement_ratio = 0.08
        
    def classify_risk_state(self, on_time_payment_rate):
        """Classify customers into risk states"""
        if on_time_payment_rate >= 95:
            return 'Prime'
        elif on_time_payment_rate >= 85:
            return 'Near-Prime'
        else:
            return 'Subprime'
    
    def calculate_expected_profit(self, loan_amount, risk_state, action):
        """Calculate expected profit for granting/denying increase"""
        if action == 0:  # Deny increase
            return 0
        
        default_prob = self.default_probs[risk_state]
        expected_profit = self.increase_profit * (1 - default_prob)
        expected_loss = loan_amount * self.default_cost_multiplier * default_prob
        
        return expected_profit - expected_loss
    
    def monte_carlo_simulation(self, n_simulations=1000, time_horizon=12):
        """Run Monte Carlo simulation for loan lifecycle"""
        results = []
        
        for _ in range(n_simulations):
             total_profit = 0
            current_capital = 1000000  
            increases_granted = 0
            
            for month in range(time_horizon):
                # Sample customers for this period
                sample_customers = self.data.sample(min(1000, len(self.data)))
                
                for _, customer in sample_customers.iterrows():
                    risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
                    
                    # Check eligibility (60 days since last loan)
                    if customer['Days Since Last Loan'] >= 60 and increases_granted < self.max_increases_per_year:
                        
                        # Calculate expected profit
                        expected_profit = self.calculate_expected_profit(
                            customer['Initial Loan ($)'], risk_state, 1
                        )
                        
                        # Grant increase if profitable and capital available
                        capital_required = customer['Initial Loan ($)'] * self.capital_requirement_ratio
                        
                        if expected_profit > 0 and current_capital >= capital_required:
                            total_profit += expected_profit
                            current_capital -= capital_required
                            increases_granted += 1
                            current_state = risk_state
                            next_state_probs = self.transition_probs[current_state]
                            next_state = np.random.choice(
                                list(next_state_probs.keys()),
                                p=list(next_state_probs.values())
                            )
            
            # Discount future profits
            discounted_profit = total_profit / ((1 + self.discount_rate/12) ** time_horizon)
            results.append(discounted_profit)
        
        return np.array(results)
                            
                            # Simulate state transition
            # Initialize simulation paramet# Cost
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LoanLimitOptimizer:
    def __init__(self, df, annual_discount_rate=0.19):
        self.data = df
        self.discount_rate = annual_discount_rate
        self.risk_states = ['Prime', 'Near-Prime', 'Subprime']
        self.setup_parameters()
        
    def setup_parameters(self):
        """Initialize model parameters"""
        # Transition probabilities (estimated from industry data)
        self.transition_probs = {
            'Prime': {'Prime': 0.85, 'Near-Prime': 0.12, 'Subprime': 0.03},
            'Near-Prime': {'Prime': 0.15, 'Near-Prime': 0.70, 'Subprime': 0.15},
            'Subprime': {'Prime': 0.05, 'Near-Prime': 0.25, 'Subprime': 0.70}
        }
        
        # Default probabilities by risk state
        self.default_probs = {'Prime': 0.01, 'Near-Prime': 0.05, 'Subprime': 0.15}
        
        # Profit parameters
        self.increase_profit = 40
        self.default_cost_multiplier = 3  
        self.max_increases_per_year = 6
        self.capital_requirement_ratio = 0.08
        
    def classify_risk_state(self, on_time_payment_rate):
        """Classify customers into risk states"""
        if on_time_payment_rate >= 95:
            return 'Prime'
        elif on_time_payment_rate >= 85:
            return 'Near-Prime'
        else:
            return 'Subprime'
    
    def calculate_expected_profit(self, loan_amount, risk_state, action):
        """Calculate expected profit for granting/denying increase"""
        if action == 0:  # Deny increase
            return 0
        
        default_prob = self.default_probs[risk_state]
        expected_profit = self.increase_profit * (1 - default_prob)
        expected_loss = loan_amount * self.default_cost_multiplier * default_prob
        
        return expected_profit - expected_loss
    
    def monte_carlo_simulation(self, n_simulations=1000, time_horizon=12):
        """Run Monte Carlo simulation for loan lifecycle"""
        results = []
        
        for _ in range(n_simulations):
             total_profit = 0
            current_capital = 1000000  
            increases_granted = 0
            
            for month in range(time_horizon):
                # Sample customers for this period
                sample_customers = self.data.sample(min(1000, len(self.data)))
                
                for _, customer in sample_customers.iterrows():
                    risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
                    
                    # Check eligibility (60 days since last loan)
                    if customer['Days Since Last Loan'] >= 60 and increases_granted < self.max_increases_per_year:
                        
                        # Calculate expected profit
                        expected_profit = self.calculate_expected_profit(
                            customer['Initial Loan ($)'], risk_state, 1
                        )
                        
                        # Grant increase if profitable and capital available
                        capital_required = customer['Initial Loan ($)'] * self.capital_requirement_ratio
                        
                        if expected_profit > 0 and current_capital >= capital_required:
                            total_profit += expected_profit
                            current_capital -= capital_required
                            increases_granted += 1
                            current_state = risk_state
                            next_state_probs = self.transition_probs[current_state]
                            next_state = np.random.choice(
                                list(next_state_probs.keys()),
                                p=list(next_state_probs.values())
                            )
            
            # Discount future profits
            discounted_profit = total_profit / ((1 + self.discount_rate/12) ** time_horizon)
            results.append(discounted_profit)
        
        return np.array(results)
                            
                            # Simulate state transition
            # Initialize simulation paramet# Cost
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LoanLimitOptimizer:
    def __init__(self, df, annual_discount_rate=0.19):
        self.data = df
        self.discount_rate = annual_discount_rate
        self.risk_states = ['Prime', 'Near-Prime', 'Subprime']
        self.setup_parameters()
        
    def setup_parameters(self):
        """Initialize model parameters"""
        # Transition probabilities (estimated from industry data)
        self.transition_probs = {
            'Prime': {'Prime': 0.85, 'Near-Prime': 0.12, 'Subprime': 0.03},
            'Near-Prime': {'Prime': 0.15, 'Near-Prime': 0.70, 'Subprime': 0.15},
            'Subprime': {'Prime': 0.05, 'Near-Prime': 0.25, 'Subprime': 0.70}
        }
        
        # Default probabilities by risk state
        self.default_probs = {'Prime': 0.01, 'Near-Prime': 0.05, 'Subprime': 0.15}
        
        # Profit parameters
        self.increase_profit = 40
        self.default_cost_multiplier = 3  # Cost of default as multiple of loan amount
        
        # Regulatory constraints
        self.max_increases_per_year = 6
        self.capital_requirement_ratio = 0.08
        
    def classify_risk_state(self, on_time_payment_rate):
        """Classify customers into risk states"""
        if on_time_payment_rate >= 95:
            return 'Prime'
        elif on_time_payment_rate >= 85:
            return 'Near-Prime'
        else:
            return 'Subprime'
    
    def calculate_expected_profit(self, loan_amount, risk_state, action):
        """Calculate expected profit for granting/denying increase"""
        if action == 0:  # Deny increase
            return 0
        
        default_prob = self.default_probs[risk_state]
        expected_profit = self.increase_profit * (1 - default_prob)
        expected_loss = loan_amount * self.default_cost_multiplier * default_prob
        
        return expected_profit - expected_loss
    
    def monte_carlo_simulation(self, n_simulations=1000, time_horizon=12):
        """Run Monte Carlo simulation for loan lifecycle"""
        results = []
        
        for _ in range(n_simulations):
            # Initialize simulation parameters
            total_profit = 0
            current_capital = 1000000  # Initial capital
            increases_granted = 0
            
            for month in range(time_horizon):
                # Sample customers for this period
                sample_customers = self.data.sample(min(1000, len(self.data)))
                
                for _, customer in sample_customers.iterrows():
                    risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
                    
                    # Check eligibility (60 days since last loan)
                    if customer['Days Since Last Loan'] >= 60 and increases_granted < self.max_increases_per_year:
                        
                        # Calculate expected profit
                        expected_profit = self.calculate_expected_profit(
                            customer['Initial Loan ($)'], risk_state, 1
                        )
                        
                        # Grant increase if profitable and capital available
                        capital_required = customer['Initial Loan ($)'] * self.capital_requirement_ratio
                        
                        if expected_profit > 0 and current_capital >= capital_required:
                            total_profit += expected_profit
                            current_capital -= capital_required
                            increases_granted += 1
                            
                            # Simulate state transition
                            current_state = risk_state
                            next_state_probs = self.transition_probs[current_state]
                            next_state = np.random.choice(
                                list(next_state_probs.keys()),
                                p=list(next_state_probs.values())
                            )
            
            # Discount future profits
            discounted_profit = total_profit / ((1 + self.discount_rate/12) ** time_horizon)
            results.append(discounted_profit)
        
        return np.array(results)
    
    def linear_programming_optimization(self):
        """Formulate and solve as linear programming problem"""
        n_customers = len(self.data)
        
        # Objective function coefficients (negative for minimization)
        c = np.zeros(n_customers)
        
        for i, (_, customer) in enumerate(self.data.iterrows()):
            risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
            expected_profit = self.calculate_expected_profit(
                customer['Initial Loan ($)'], risk_state, 1
            )
            c[i] = -expected_profit  # Negative for minimization
            
        # Constraints: Capital requirement
        capital_coefficients = [
            customer['Initial Loan ($)'] * self.capital_requirement_ratio 
            for _, customer in self.data.iterrows()
        ]
        
        A_ub = [capital_coefficients]  # Capital constraint
        b_ub = [500000]  # Maximum capital available
        
        # Bounds: 0 <= x_i <= 1 (fraction of increase granted)
        bounds = [(0, 1) for _ in range(n_customers)]
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        return result
    
    def reinforcement_learning_policy(self, n_episodes=1000):
        """Implement Q-learning for optimal policy"""
        # Simplified Q-learning implementation
        n_states = len(self.risk_states)
        n_actions = 2
        
        # Initialize Q-table
        Q = np.zeros((n_states, n_actions))
        
        # Learning parameters
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.1  # Exploration rate
        
        for episode in range(n_episodes):
            # Start with random state
            state_idx = np.random.randint(0, n_states)
            total_reward = 0
            
            for step in range(10):  # 10 steps per episode
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(0, n_actions)
                else:
                    action = np.argmax(Q[state_idx])
                
                # Simulate reward (simplified)
                state = self.risk_states[state_idx]
                reward = self.calculate_expected_profit(1000, state, action)
                
                # Next state (simplified transition)
                next_state_idx = np.random.randint(0, n_states)
                
                # Update Q-value
                Q[state_idx, action] = (1 - alpha) * Q[state_idx, action] + \
                                      alpha * (reward + gamma * np.max(Q[next_state_idx]))
                
                state_idx = next_state_idx
                total_reward += reward
        
        return Q
    
    def demand_forecasting(self, economic_factors=None):
        """Forecast customer uptake of loan increases"""
        if economic_factors is None:
            economic_factors = {
                'inflation_rate': 0.03,
                'unemployment_rate': 0.05,
                'interest_rate': 0.07
            }
        
        # Feature engineering for demand prediction
        self.data['Risk_State'] = self.data['On-time Payments (%)'].apply(self.classify_risk_state)
        self.data['Days_Since_Last_Category'] = pd.cut(
            self.data['Days Since Last Loan'], 
            bins=[0, 30, 60, 90, 180, 365, float('inf')]
        )
        
        # Simplified demand model
        base_acceptance_rate = 0.6
        economic_impact = 1 - (economic_factors['unemployment_rate'] * 0.5 + 
                              economic_factors['inflation_rate'] * 0.3)
        
        risk_adjustment = {
            'Prime': 1.2,
            'Near-Prime': 1.0,
            'Subprime': 0.7
        }
        
        self.data['Predicted_Uptake'] = self.data['Risk_State'].map(risk_adjustment) * \
                                      base_acceptance_rate * economic_impact
        
        return self.data['Predicted_Uptake'].mean()
    
    def optimize_strategy(self):
        """Main optimization method combining all approaches"""
        print("Starting loan limit optimization...")
        
        # 1. Demand forecasting
        avg_uptake = self.demand_forecasting()
        print(f"Predicted average uptake rate: {avg_uptake:.2%}")
        
        # 2. Monte Carlo simulation
        print("Running Monte Carlo simulation...")
        mc_results = self.monte_carlo_simulation(n_simulations=500)
        print(f"Expected profit distribution: Mean=${mc_results.mean():.2f}, Std=${mc_results.std():.2f}")
        
        # 3. Linear programming optimization
        print("Solving linear programming problem...")
        lp_result = self.linear_programming_optimization()
        print(f"LP optimization status: {lp_result.status}")
        print(f"Optimal objective value: ${-lp_result.fun:.2f}")
        
        # 4. Reinforcement learning policy
        print("Training reinforcement learning policy...")
        q_table = self.reinforcement_learning_policy(n_episodes=500)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(q_table)
        
        return {
            'monte_carlo_results': mc_results,
            'lp_solution': lp_result,
            'q_table': q_table,
            'recommendations': recommendations,
            'demand_forecast': avg_uptake
        }
    
    def generate_recommendations(self, q_table):
        """Generate actionable recommendations based on optimization results"""
        recommendations = []
        
        for _, customer in self.data.iterrows():
            risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
            state_idx = self.risk_states.index(risk_state)
            
            # Get optimal action from Q-table
            optimal_action = np.argmax(q_table[state_idx])
            
            recommendation = {
                'Customer_ID': customer['Customer ID'],
                'Risk_State': risk_state,
                'Recommended_Action': 'GRANT' if optimal_action == 1 else 'DENY',
                'Expected_Profit': self.calculate_expected_profit(
                    customer['Initial Loan ($)'], risk_state, optimal_action
                ),
                'Days_Since_Last_Loan': customer['Days Since Last Loan'],
                'On_Time_Payments': customer['On-time Payments (%)']
            }
            recommendations.append(recommendation)
        
        return pd.DataFrame(recommendations)

# Implementation and Analysis
def main():
    # Load and prepare data
    print("Loading dataset...")
    df = pd.read_csv('loan_limit_increases.csv')
    
    # Initialize optimizer
    optimizer = LoanLimitOptimizer(df)
    
    # Run comprehensive optimization
    results = optimizer.optimize_strategy()
    
    # Analyze results
    analyze_results(results, df)
    
    # Generate strategic insights
    generate_strategic_insights(results, df)

def analyze_results(results, df):
    """Analyze and visualize optimization results"""
    
    # 1. Profit distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results['monte_carlo_results'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Monte Carlo Simulation: Profit Distribution')
    plt.xlabel('Profit ($)')
    plt.ylabel('Frequency')
    
    # 2. Risk state distribution
    plt.subplot(2, 2, 2)
    risk_states = df['On-time Payments (%)'].apply(optimizer.classify_risk_state)
    risk_counts = risk_states.value_counts()
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
    plt.title('Customer Risk State Distribution')
    
    # 3. Action recommendations by risk state
    plt.subplot(2, 2, 3)
    recommendations = results['recommendations']
    action_by_risk = pd.crosstab(recommendations['Risk_State'], recommendations['Recommended_Action'])
    action_by_risk.plot(kind='bar', ax=plt.gca())
    plt.title('Recommended Actions by Risk State')
    plt.xlabel('Risk State')
    plt.ylabel('Count')
    plt.legend(title='Action')
    
    # 4. Expected profit by risk state
    plt.subplot(2, 2, 4)
    profit_by_risk = recommendations.groupby('Risk_State')['Expected_Profit'].mean()
    plt.bar(profit_by_risk.index, profit_by_risk.values)
    plt.title('Average Expected Profit by Risk State')
    plt.xlabel('Risk State')
    plt.ylabel('Expected Profit ($)')
    
    plt.tight_layout()
    plt.show()

def generate_strategic_insights(results, df):
    """Generate strategic business insights"""
    
    recommendations = results['recommendations']
    
    print("\n" + "="*60)
    print("STRATEGIC INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    # 1. Overall strategy
    total_customers = len(recommendations)
    grant_recommendations = len(recommendations[recommendations['Recommended_Action'] == 'GRANT'])
    grant_rate = grant_recommendations / total_customers
    
    print(f"\n1. OVERALL STRATEGY:")
    print(f"   - Recommended grant rate: {grant_rate:.1%}")
    print(f"   - Total customers analyzed: {total_customers:,}")
    print(f"   - Customers recommended for increase: {grant_recommendations:,}")
    
    # 2. Risk-based insights
    print(f"\n2. RISK-BASED INSIGHTS:")
    for risk_state in ['Prime', 'Near-Prime', 'Subprime']:
        risk_customers = recommendations[recommendations['Risk_State'] == risk_state]
        risk_grant_rate = len(risk_customers[risk_customers['Recommended_Action'] == 'GRANT']) / len(risk_customers)
        avg_profit = risk_customers['Expected_Profit'].mean()
        
        print(f"   - {risk_state}:")
        print(f"     * Grant rate: {risk_grant_rate:.1%}")
        print(f"     * Avg expected profit: ${avg_profit:.2f}")
    
    # 3. Economic sensitivity
    print(f"\n3. ECONOMIC SENSITIVITY ANALYSIS:")
    economic_scenarios = [
        {'inflation_rate': 0.02, 'unemployment_rate': 0.03, 'interest_rate': 0.05},  # Boom
        {'inflation_rate': 0.03, 'unemployment_rate': 0.05, 'interest_rate': 0.07},  # Normal
        {'inflation_rate': 0.06, 'unemployment_rate': 0.08, 'interest_rate': 0.10},  # Recession
    ]
    
    scenario_names = ['Economic Boom', 'Normal Conditions', 'Recession']
    
    for scenario, name in zip(economic_scenarios, scenario_names):
        uptake_rate = optimizer.demand_forecasting(scenario)
        print(f"   - {name}: Predicted uptake rate = {uptake_rate:.1%}")
    
    # 4. Implementation recommendations
    print(f"\n4. IMPLEMENTATION RECOMMENDATIONS:")
    print("   - Implement dynamic risk-based pricing for limit increases")
    print("   - Use reinforcement learning for real-time policy adaptation")
    print("   - Monitor economic indicators for strategy adjustments")
    print("   - Implement behavioral nudges for high-risk customers")
    print("   - Establish regular model retraining schedule (quarterly)")

# Run the complete analysis
if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LoanLimitOptimizer:
    def __init__(self, df, annual_discount_rate=0.19):
        self.data = df
        self.discount_rate = annual_discount_rate
        self.risk_states = ['Prime', 'Near-Prime', 'Subprime']
        self.setup_parameters()
        
    def setup_parameters(self):
        """Initialize model parameters"""
        # Transition probabilities (estimated from industry data)
        self.transition_probs = {
            'Prime': {'Prime': 0.85, 'Near-Prime': 0.12, 'Subprime': 0.03},
            'Near-Prime': {'Prime': 0.15, 'Near-Prime': 0.70, 'Subprime': 0.15},
            'Subprime': {'Prime': 0.05, 'Near-Prime': 0.25, 'Subprime': 0.70}
        }
        
        # Default probabilities by risk state
        self.default_probs = {'Prime': 0.01, 'Near-Prime': 0.05, 'Subprime': 0.15}
        
        # Profit parameters
        self.increase_profit = 40
        self.default_cost_multiplier = 3  # Cost of default as multiple of loan amount
        
        # Regulatory constraints
        self.max_increases_per_year = 6
        self.capital_requirement_ratio = 0.08
        
    def classify_risk_state(self, on_time_payment_rate):
        """Classify customers into risk states"""
        if on_time_payment_rate >= 95:
            return 'Prime'
        elif on_time_payment_rate >= 85:
            return 'Near-Prime'
        else:
            return 'Subprime'
    
    def calculate_expected_profit(self, loan_amount, risk_state, action):
        """Calculate expected profit for granting/denying increase"""
        if action == 0:  # Deny increase
            return 0
        
        default_prob = self.default_probs[risk_state]
        expected_profit = self.increase_profit * (1 - default_prob)
        expected_loss = loan_amount * self.default_cost_multiplier * default_prob
        
        return expected_profit - expected_loss
    
    def monte_carlo_simulation(self, n_simulations=1000, time_horizon=12):
        """Run Monte Carlo simulation for loan lifecycle"""
        results = []
        
        for _ in range(n_simulations):
            # Initialize simulation parameters
            total_profit = 0
            current_capital = 1000000  # Initial capital
            increases_granted = 0
            
            for month in range(time_horizon):
                # Sample customers for this period
                sample_customers = self.data.sample(min(1000, len(self.data)))
                
                for _, customer in sample_customers.iterrows():
                    risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
                    
                    # Check eligibility (60 days since last loan)
                    if customer['Days Since Last Loan'] >= 60 and increases_granted < self.max_increases_per_year:
                        
                        # Calculate expected profit
                        expected_profit = self.calculate_expected_profit(
                            customer['Initial Loan ($)'], risk_state, 1
                        )
                        
                        # Grant increase if profitable and capital available
                        capital_required = customer['Initial Loan ($)'] * self.capital_requirement_ratio
                        
                        if expected_profit > 0 and current_capital >= capital_required:
                            total_profit += expected_profit
                            current_capital -= capital_required
                            increases_granted += 1
                            
                            # Simulate state transition
                            current_state = risk_state
                            next_state_probs = self.transition_probs[current_state]
                            next_state = np.random.choice(
                                list(next_state_probs.keys()),
                                p=list(next_state_probs.values())
                            )
            
            # Discount future profits
            discounted_profit = total_profit / ((1 + self.discount_rate/12) ** time_horizon)
            results.append(discounted_profit)
        
        return np.array(results)
    
    def linear_programming_optimization(self):
        """Formulate and solve as linear programming problem"""
        n_customers = len(self.data)
        
        # Objective function coefficients (negative for minimization)
        c = np.zeros(n_customers)
        
        for i, (_, customer) in enumerate(self.data.iterrows()):
            risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
            expected_profit = self.calculate_expected_profit(
                customer['Initial Loan ($)'], risk_state, 1
            )
            c[i] = -expected_profit  # Negative for minimization
            
        # Constraints: Capital requirement
        capital_coefficients = [
            customer['Initial Loan ($)'] * self.capital_requirement_ratio 
            for _, customer in self.data.iterrows()
        ]
        
        A_ub = [capital_coefficients]  # Capital constraint
        b_ub = [500000]  # Maximum capital available
        
        # Bounds: 0 <= x_i <= 1 (fraction of increase granted)
        bounds = [(0, 1) for _ in range(n_customers)]
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        return result
    
    def reinforcement_learning_policy(self, n_episodes=1000):
        """Implement Q-learning for optimal policy"""
        # Simplified Q-learning implementation
        n_states = len(self.risk_states)
        n_actions = 2
        
        # Initialize Q-table
        Q = np.zeros((n_states, n_actions))
        
        # Learning parameters
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.1  # Exploration rate
        
        for episode in range(n_episodes):
            # Start with random state
            state_idx = np.random.randint(0, n_states)
            total_reward = 0
            
            for step in range(10):  # 10 steps per episode
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(0, n_actions)
                else:
                    action = np.argmax(Q[state_idx])
                
                # Simulate reward (simplified)
                state = self.risk_states[state_idx]
                reward = self.calculate_expected_profit(1000, state, action)
                
                # Next state (simplified transition)
                next_state_idx = np.random.randint(0, n_states)
                
                # Update Q-value
                Q[state_idx, action] = (1 - alpha) * Q[state_idx, action] + \
                                      alpha * (reward + gamma * np.max(Q[next_state_idx]))
                
                state_idx = next_state_idx
                total_reward += reward
        
        return Q
    
    def demand_forecasting(self, economic_factors=None):
        """Forecast customer uptake of loan increases"""
        if economic_factors is None:
            economic_factors = {
                'inflation_rate': 0.03,
                'unemployment_rate': 0.05,
                'interest_rate': 0.07
            }
        
        # Feature engineering for demand prediction
        self.data['Risk_State'] = self.data['On-time Payments (%)'].apply(self.classify_risk_state)
        self.data['Days_Since_Last_Category'] = pd.cut(
            self.data['Days Since Last Loan'], 
            bins=[0, 30, 60, 90, 180, 365, float('inf')]
        )
        
        # Simplified demand model
        base_acceptance_rate = 0.6
        economic_impact = 1 - (economic_factors['unemployment_rate'] * 0.5 + 
                              economic_factors['inflation_rate'] * 0.3)
        
        risk_adjustment = {
            'Prime': 1.2,
            'Near-Prime': 1.0,
            'Subprime': 0.7
        }
        
        self.data['Predicted_Uptake'] = self.data['Risk_State'].map(risk_adjustment) * \
                                      base_acceptance_rate * economic_impact
        
        return self.data['Predicted_Uptake'].mean()
    
    def optimize_strategy(self):
        """Main optimization method combining all approaches"""
        print("Starting loan limit optimization...")
        
        # 1. Demand forecasting
        avg_uptake = self.demand_forecasting()
        print(f"Predicted average uptake rate: {avg_uptake:.2%}")
        
        # 2. Monte Carlo simulation
        print("Running Monte Carlo simulation...")
        mc_results = self.monte_carlo_simulation(n_simulations=500)
        print(f"Expected profit distribution: Mean=${mc_results.mean():.2f}, Std=${mc_results.std():.2f}")
        
        # 3. Linear programming optimization
        print("Solving linear programming problem...")
        lp_result = self.linear_programming_optimization()
        print(f"LP optimization status: {lp_result.status}")
        print(f"Optimal objective value: ${-lp_result.fun:.2f}")
        
        # 4. Reinforcement learning policy
        print("Training reinforcement learning policy...")
        q_table = self.reinforcement_learning_policy(n_episodes=500)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(q_table)
        
        return {
            'monte_carlo_results': mc_results,
            'lp_solution': lp_result,
            'q_table': q_table,
            'recommendations': recommendations,
            'demand_forecast': avg_uptake
        }
    
    def generate_recommendations(self, q_table):
        """Generate actionable recommendations based on optimization results"""
        recommendations = []
        
        for _, customer in self.data.iterrows():
            risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
            state_idx = self.risk_states.index(risk_state)
            
            # Get optimal action from Q-table
            optimal_action = np.argmax(q_table[state_idx])
            
            recommendation = {
                'Customer_ID': customer['Customer ID'],
                'Risk_State': risk_state,
                'Recommended_Action': 'GRANT' if optimal_action == 1 else 'DENY',
                'Expected_Profit': self.calculate_expected_profit(
                    customer['Initial Loan ($)'], risk_state, optimal_action
                ),
                'Days_Since_Last_Loan': customer['Days Since Last Loan'],
                'On_Time_Payments': customer['On-time Payments (%)']
            }
            recommendations.append(recommendation)
        
        return pd.DataFrame(recommendations)

# Implementation and Analysis
def main():
    # Load and prepare data
    print("Loading dataset...")
    df = pd.read_excel(C:\Users\Administrator\Downloads\loan_limit_increases.xlsx")
    
    # Initialize optimizer
    optimizer = LoanLimitOptimizer(df)
    
    # Run comprehensive optimization
    results = optimizer.optimize_strategy()
    
    # Analyze results
    analyze_results(results, df)
    
    # Generate strategic insights
    generate_strategic_insights(results, df)

def analyze_results(results, df):
    """Analyze and visualize optimization results"""
    
    # 1. Profit distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results['monte_carlo_results'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Monte Carlo Simulation: Profit Distribution')
    plt.xlabel('Profit ($)')
    plt.ylabel('Frequency')
    
    # 2. Risk state distribution
    plt.subplot(2, 2, 2)
    risk_states = df['On-time Payments (%)'].apply(optimizer.classify_risk_state)
    risk_counts = risk_states.value_counts()
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
    plt.title('Customer Risk State Distribution')
    
    # 3. Action recommendations by risk state
    plt.subplot(2, 2, 3)
    recommendations = results['recommendations']
    action_by_risk = pd.crosstab(recommendations['Risk_State'], recommendations['Recommended_Action'])
    action_by_risk.plot(kind='bar', ax=plt.gca())
    plt.title('Recommended Actions by Risk State')
    plt.xlabel('Risk State')
    plt.ylabel('Count')
    plt.legend(title='Action')
    
    # 4. Expected profit by risk state
    plt.subplot(2, 2, 4)
    profit_by_risk = recommendations.groupby('Risk_State')['Expected_Profit'].mean()
    plt.bar(profit_by_risk.index, profit_by_risk.values)
    plt.title('Average Expected Profit by Risk State')
    plt.xlabel('Risk State')
    plt.ylabel('Expected Profit ($)')
    
    plt.tight_layout()
    plt.show()

def generate_strategic_insights(results, df):
    """Generate strategic business insights"""
    
    recommendations = results['recommendations']
    
    print("\n" + "="*60)
    print("STRATEGIC INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    # 1. Overall strategy
    total_customers = len(recommendations)
    grant_recommendations = len(recommendations[recommendations['Recommended_Action'] == 'GRANT'])
    grant_rate = grant_recommendations / total_customers
    
    print(f"\n1. OVERALL STRATEGY:")
    print(f"   - Recommended grant rate: {grant_rate:.1%}")
    print(f"   - Total customers analyzed: {total_customers:,}")
    print(f"   - Customers recommended for increase: {grant_recommendations:,}")
    
    # 2. Risk-based insights
    print(f"\n2. RISK-BASED INSIGHTS:")
    for risk_state in ['Prime', 'Near-Prime', 'Subprime']:
        risk_customers = recommendations[recommendations['Risk_State'] == risk_state]
        risk_grant_rate = len(risk_customers[risk_customers['Recommended_Action'] == 'GRANT']) / len(risk_customers)
        avg_profit = risk_customers['Expected_Profit'].mean()
        
        print(f"   - {risk_state}:")
        print(f"     * Grant rate: {risk_grant_rate:.1%}")
        print(f"     * Avg expected profit: ${avg_profit:.2f}")
    
    # 3. Economic sensitivity
    print(f"\n3. ECONOMIC SENSITIVITY ANALYSIS:")
    economic_scenarios = [
        {'inflation_rate': 0.02, 'unemployment_rate': 0.03, 'interest_rate': 0.05},  # Boom
        {'inflation_rate': 0.03, 'unemployment_rate': 0.05, 'interest_rate': 0.07},  # Normal
        {'inflation_rate': 0.06, 'unemployment_rate': 0.08, 'interest_rate': 0.10},  # Recession
    ]
    
    scenario_names = ['Economic Boom', 'Normal Conditions', 'Recession']
    
    for scenario, name in zip(economic_scenarios, scenario_names):
        uptake_rate = optimizer.demand_forecasting(scenario)
        print(f"   - {name}: Predicted uptake rate = {uptake_rate:.1%}")
    
    # 4. Implementation recommendations
    print(f"\n4. IMPLEMENTATION RECOMMENDATIONS:")
    print("   - Implement dynamic risk-based pricing for limit increases")
    print("   - Use reinforcement learning for real-time policy adaptation")
    print("   - Monitor economic indicators for strategy adjustments")
    print("   - Implement behavioral nudges for high-risk customers")
    print("   - Establish regular model retraining schedule (quarterly)")

# Run the complete analysis
if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LoanLimitOptimizer:
    def __init__(self, df, annual_discount_rate=0.19):
        self.data = df
        self.discount_rate = annual_discount_rate
        self.risk_states = ['Prime', 'Near-Prime', 'Subprime']
        self.setup_parameters()
        
    def setup_parameters(self):
        """Initialize model parameters"""
        # Transition probabilities (estimated from industry data)
        self.transition_probs = {
            'Prime': {'Prime': 0.85, 'Near-Prime': 0.12, 'Subprime': 0.03},
            'Near-Prime': {'Prime': 0.15, 'Near-Prime': 0.70, 'Subprime': 0.15},
            'Subprime': {'Prime': 0.05, 'Near-Prime': 0.25, 'Subprime': 0.70}
        }
        
        # Default probabilities by risk state
        self.default_probs = {'Prime': 0.01, 'Near-Prime': 0.05, 'Subprime': 0.15}
        
        # Profit parameters
        self.increase_profit = 40
        self.default_cost_multiplier = 3  # Cost of default as multiple of loan amount
        
        # Regulatory constraints
        self.max_increases_per_year = 6
        self.capital_requirement_ratio = 0.08
        
    def classify_risk_state(self, on_time_payment_rate):
        """Classify customers into risk states"""
        if on_time_payment_rate >= 95:
            return 'Prime'
        elif on_time_payment_rate >= 85:
            return 'Near-Prime'
        else:
            return 'Subprime'
    
    def calculate_expected_profit(self, loan_amount, risk_state, action):
        """Calculate expected profit for granting/denying increase"""
        if action == 0:  # Deny increase
            return 0
        
        default_prob = self.default_probs[risk_state]
        expected_profit = self.increase_profit * (1 - default_prob)
        expected_loss = loan_amount * self.default_cost_multiplier * default_prob
        
        return expected_profit - expected_loss
    
    def monte_carlo_simulation(self, n_simulations=1000, time_horizon=12):
        """Run Monte Carlo simulation for loan lifecycle"""
        results = []
        
        for _ in range(n_simulations):
            # Initialize simulation parameters
            total_profit = 0
            current_capital = 1000000  # Initial capital
            increases_granted = 0
            
            for month in range(time_horizon):
                # Sample customers for this period
                sample_customers = self.data.sample(min(1000, len(self.data)))
                
                for _, customer in sample_customers.iterrows():
                    risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
                    
                    # Check eligibility (60 days since last loan)
                    if customer['Days Since Last Loan'] >= 60 and increases_granted < self.max_increases_per_year:
                        
                        # Calculate expected profit
                        expected_profit = self.calculate_expected_profit(
                            customer['Initial Loan ($)'], risk_state, 1
                        )
                        
                        # Grant increase if profitable and capital available
                        capital_required = customer['Initial Loan ($)'] * self.capital_requirement_ratio
                        
                        if expected_profit > 0 and current_capital >= capital_required:
                            total_profit += expected_profit
                            current_capital -= capital_required
                            increases_granted += 1
                            
                            # Simulate state transition
                            current_state = risk_state
                            next_state_probs = self.transition_probs[current_state]
                            next_state = np.random.choice(
                                list(next_state_probs.keys()),
                                p=list(next_state_probs.values())
                            )
            
            # Discount future profits
            discounted_profit = total_profit / ((1 + self.discount_rate/12) ** time_horizon)
            results.append(discounted_profit)
        
        return np.array(results)
    
    def linear_programming_optimization(self):
        """Formulate and solve as linear programming problem"""
        n_customers = len(self.data)
        
        # Objective function coefficients (negative for minimization)
        c = np.zeros(n_customers)
        
        for i, (_, customer) in enumerate(self.data.iterrows()):
            risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
            expected_profit = self.calculate_expected_profit(
                customer['Initial Loan ($)'], risk_state, 1
            )
            c[i] = -expected_profit  # Negative for minimization
            
        # Constraints: Capital requirement
        capital_coefficients = [
            customer['Initial Loan ($)'] * self.capital_requirement_ratio 
            for _, customer in self.data.iterrows()
        ]
        
        A_ub = [capital_coefficients]  # Capital constraint
        b_ub = [500000]  # Maximum capital available
        
        # Bounds: 0 <= x_i <= 1 (fraction of increase granted)
        bounds = [(0, 1) for _ in range(n_customers)]
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        return result
    
    def reinforcement_learning_policy(self, n_episodes=1000):
        """Implement Q-learning for optimal policy"""
        # Simplified Q-learning implementation
        n_states = len(self.risk_states)
        n_actions = 2
        
        # Initialize Q-table
        Q = np.zeros((n_states, n_actions))
        
        # Learning parameters
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.1  # Exploration rate
        
        for episode in range(n_episodes):
            # Start with random state
            state_idx = np.random.randint(0, n_states)
            total_reward = 0
            
            for step in range(10):  # 10 steps per episode
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(0, n_actions)
                else:
                    action = np.argmax(Q[state_idx])
                
                # Simulate reward (simplified)
                state = self.risk_states[state_idx]
                reward = self.calculate_expected_profit(1000, state, action)
                
                # Next state (simplified transition)
                next_state_idx = np.random.randint(0, n_states)
                
                # Update Q-value
                Q[state_idx, action] = (1 - alpha) * Q[state_idx, action] + \
                                      alpha * (reward + gamma * np.max(Q[next_state_idx]))
                
                state_idx = next_state_idx
                total_reward += reward
        
        return Q
    
    def demand_forecasting(self, economic_factors=None):
        """Forecast customer uptake of loan increases"""
        if economic_factors is None:
            economic_factors = {
                'inflation_rate': 0.03,
                'unemployment_rate': 0.05,
                'interest_rate': 0.07
            }
        
        # Feature engineering for demand prediction
        self.data['Risk_State'] = self.data['On-time Payments (%)'].apply(self.classify_risk_state)
        self.data['Days_Since_Last_Category'] = pd.cut(
            self.data['Days Since Last Loan'], 
            bins=[0, 30, 60, 90, 180, 365, float('inf')]
        )
        
        # Simplified demand model
        base_acceptance_rate = 0.6
        economic_impact = 1 - (economic_factors['unemployment_rate'] * 0.5 + 
                              economic_factors['inflation_rate'] * 0.3)
        
        risk_adjustment = {
            'Prime': 1.2,
            'Near-Prime': 1.0,
            'Subprime': 0.7
        }
        
        self.data['Predicted_Uptake'] = self.data['Risk_State'].map(risk_adjustment) * \
                                      base_acceptance_rate * economic_impact
        
        return self.data['Predicted_Uptake'].mean()
    
    def optimize_strategy(self):
        """Main optimization method combining all approaches"""
        print("Starting loan limit optimization...")
        
        # 1. Demand forecasting
        avg_uptake = self.demand_forecasting()
        print(f"Predicted average uptake rate: {avg_uptake:.2%}")
        
        # 2. Monte Carlo simulation
        print("Running Monte Carlo simulation...")
        mc_results = self.monte_carlo_simulation(n_simulations=500)
        print(f"Expected profit distribution: Mean=${mc_results.mean():.2f}, Std=${mc_results.std():.2f}")
        
        # 3. Linear programming optimization
        print("Solving linear programming problem...")
        lp_result = self.linear_programming_optimization()
        print(f"LP optimization status: {lp_result.status}")
        print(f"Optimal objective value: ${-lp_result.fun:.2f}")
        
        # 4. Reinforcement learning policy
        print("Training reinforcement learning policy...")
        q_table = self.reinforcement_learning_policy(n_episodes=500)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(q_table)
        
        return {
            'monte_carlo_results': mc_results,
            'lp_solution': lp_result,
            'q_table': q_table,
            'recommendations': recommendations,
            'demand_forecast': avg_uptake
        }
    
    def generate_recommendations(self, q_table):
        """Generate actionable recommendations based on optimization results"""
        recommendations = []
        
        for _, customer in self.data.iterrows():
            risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
            state_idx = self.risk_states.index(risk_state)
            
            # Get optimal action from Q-table
            optimal_action = np.argmax(q_table[state_idx])
            
            recommendation = {
                'Customer_ID': customer['Customer ID'],
                'Risk_State': risk_state,
                'Recommended_Action': 'GRANT' if optimal_action == 1 else 'DENY',
                'Expected_Profit': self.calculate_expected_profit(
                    customer['Initial Loan ($)'], risk_state, optimal_action
                ),
                'Days_Since_Last_Loan': customer['Days Since Last Loan'],
                'On_Time_Payments': customer['On-time Payments (%)']
            }
            recommendations.append(recommendation)
        
        return pd.DataFrame(recommendations)

# Implementation and Analysis
def main():
    # Load and prepare data
    print("Loading dataset...")
    df = pd.read_excel(r"C:\Users\Administrator\Downloads\loan_limit_increases.xlsx")
    
    # Initialize optimizer
    optimizer = LoanLimitOptimizer(df)
    
    # Run comprehensive optimization
    results = optimizer.optimize_strategy()
    
    # Analyze results
    analyze_results(results, df)
    
    # Generate strategic insights
    generate_strategic_insights(results, df)

def analyze_results(results, df):
    """Analyze and visualize optimization results"""
    
    # 1. Profit distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results['monte_carlo_results'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Monte Carlo Simulation: Profit Distribution')
    plt.xlabel('Profit ($)')
    plt.ylabel('Frequency')
    
    # 2. Risk state distribution
    plt.subplot(2, 2, 2)
    risk_states = df['On-time Payments (%)'].apply(optimizer.classify_risk_state)
    risk_counts = risk_states.value_counts()
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
    plt.title('Customer Risk State Distribution')
    
    # 3. Action recommendations by risk state
    plt.subplot(2, 2, 3)
    recommendations = results['recommendations']
    action_by_risk = pd.crosstab(recommendations['Risk_State'], recommendations['Recommended_Action'])
    action_by_risk.plot(kind='bar', ax=plt.gca())
    plt.title('Recommended Actions by Risk State')
    plt.xlabel('Risk State')
    plt.ylabel('Count')
    plt.legend(title='Action')
    
    # 4. Expected profit by risk state
    plt.subplot(2, 2, 4)
    profit_by_risk = recommendations.groupby('Risk_State')['Expected_Profit'].mean()
    plt.bar(profit_by_risk.index, profit_by_risk.values)
    plt.title('Average Expected Profit by Risk State')
    plt.xlabel('Risk State')
    plt.ylabel('Expected Profit ($)')
    
    plt.tight_layout()
    plt.show()

def generate_strategic_insights(results, df):
    """Generate strategic business insights"""
    
    recommendations = results['recommendations']
    
    print("\n" + "="*60)
    print("STRATEGIC INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    # 1. Overall strategy
    total_customers = len(recommendations)
    grant_recommendations = len(recommendations[recommendations['Recommended_Action'] == 'GRANT'])
    grant_rate = grant_recommendations / total_customers
    
    print(f"\n1. OVERALL STRATEGY:")
    print(f"   - Recommended grant rate: {grant_rate:.1%}")
    print(f"   - Total customers analyzed: {total_customers:,}")
    print(f"   - Customers recommended for increase: {grant_recommendations:,}")
    
    # 2. Risk-based insights
    print(f"\n2. RISK-BASED INSIGHTS:")
    for risk_state in ['Prime', 'Near-Prime', 'Subprime']:
        risk_customers = recommendations[recommendations['Risk_State'] == risk_state]
        risk_grant_rate = len(risk_customers[risk_customers['Recommended_Action'] == 'GRANT']) / len(risk_customers)
        avg_profit = risk_customers['Expected_Profit'].mean()
        
        print(f"   - {risk_state}:")
        print(f"     * Grant rate: {risk_grant_rate:.1%}")
        print(f"     * Avg expected profit: ${avg_profit:.2f}")
    
    # 3. Economic sensitivity
    print(f"\n3. ECONOMIC SENSITIVITY ANALYSIS:")
    economic_scenarios = [
        {'inflation_rate': 0.02, 'unemployment_rate': 0.03, 'interest_rate': 0.05},  # Boom
        {'inflation_rate': 0.03, 'unemployment_rate': 0.05, 'interest_rate': 0.07},  # Normal
        {'inflation_rate': 0.06, 'unemployment_rate': 0.08, 'interest_rate': 0.10},  # Recession
    ]
    
    scenario_names = ['Economic Boom', 'Normal Conditions', 'Recession']
    
    for scenario, name in zip(economic_scenarios, scenario_names):
        uptake_rate = optimizer.demand_forecasting(scenario)
        print(f"   - {name}: Predicted uptake rate = {uptake_rate:.1%}")
    
    # 4. Implementation recommendations
    print(f"\n4. IMPLEMENTATION RECOMMENDATIONS:")
    print("   - Implement dynamic risk-based pricing for limit increases")
    print("   - Use reinforcement learning for real-time policy adaptation")
    print("   - Monitor economic indicators for strategy adjustments")
    print("   - Implement behavioral nudges for high-risk customers")
    print("   - Establish regular model retraining schedule (quarterly)")

# Run the complete analysis
if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LoanLimitOptimizer:
    def __init__(self, data, annual_discount_rate=0.19):
        self.data = data
        self.discount_rate = annual_discount_rate
        self.risk_states = ['Prime', 'Near-Prime', 'Subprime']
        self.setup_parameters()
        
    def setup_parameters(self):
        """Initialize model parameters"""
        # Transition probabilities (estimated from industry data)
        self.transition_probs = {
            'Prime': {'Prime': 0.85, 'Near-Prime': 0.12, 'Subprime': 0.03},
            'Near-Prime': {'Prime': 0.15, 'Near-Prime': 0.70, 'Subprime': 0.15},
            'Subprime': {'Prime': 0.05, 'Near-Prime': 0.25, 'Subprime': 0.70}
        }
        
        # Default probabilities by risk state
        self.default_probs = {'Prime': 0.01, 'Near-Prime': 0.05, 'Subprime': 0.15}
        
        # Profit parameters
        self.increase_profit = 40
        self.default_cost_multiplier = 3  # Cost of default as multiple of loan amount
        
        # Regulatory constraints
        self.max_increases_per_year = 6
        self.capital_requirement_ratio = 0.08
        
    def classify_risk_state(self, on_time_payment_rate):
        """Classify customers into risk states"""
        if on_time_payment_rate >= 95:
            return 'Prime'
        elif on_time_payment_rate >= 85:
            return 'Near-Prime'
        else:
            return 'Subprime'
    
    def calculate_expected_profit(self, loan_amount, risk_state, action):
        """Calculate expected profit for granting/denying increase"""
        if action == 0:  # Deny increase
            return 0
        
        default_prob = self.default_probs[risk_state]
        expected_profit = self.increase_profit * (1 - default_prob)
        expected_loss = loan_amount * self.default_cost_multiplier * default_prob
        
        return expected_profit - expected_loss
    
    def monte_carlo_simulation(self, n_simulations=1000, time_horizon=12):
        """Run Monte Carlo simulation for loan lifecycle"""
        results = []
        
        for _ in range(n_simulations):
            # Initialize simulation parameters
            total_profit = 0
            current_capital = 1000000  # Initial capital
            increases_granted = 0
            
            for month in range(time_horizon):
                # Sample customers for this period
                sample_customers = self.data.sample(min(1000, len(self.data)))
                
                for _, customer in sample_customers.iterrows():
                    risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
                    
                    # Check eligibility (60 days since last loan)
                    if customer['Days Since Last Loan'] >= 60 and increases_granted < self.max_increases_per_year:
                        
                        # Calculate expected profit
                        expected_profit = self.calculate_expected_profit(
                            customer['Initial Loan ($)'], risk_state, 1
                        )
                        
                        # Grant increase if profitable and capital available
                        capital_required = customer['Initial Loan ($)'] * self.capital_requirement_ratio
                        
                        if expected_profit > 0 and current_capital >= capital_required:
                            total_profit += expected_profit
                            current_capital -= capital_required
                            increases_granted += 1
                            
                            # Simulate state transition
                            current_state = risk_state
                            next_state_probs = self.transition_probs[current_state]
                            next_state = np.random.choice(
                                list(next_state_probs.keys()),
                                p=list(next_state_probs.values())
                            )
            
            # Discount future profits
            discounted_profit = total_profit / ((1 + self.discount_rate/12) ** time_horizon)
            results.append(discounted_profit)
        
        return np.array(results)
    
    def linear_programming_optimization(self):
        """Formulate and solve as linear programming problem"""
        n_customers = len(self.data)
        
        # Objective function coefficients (negative for minimization)
        c = np.zeros(n_customers)
        
        for i, (_, customer) in enumerate(self.data.iterrows()):
            risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
            expected_profit = self.calculate_expected_profit(
                customer['Initial Loan ($)'], risk_state, 1
            )
            c[i] = -expected_profit  # Negative for minimization
            
        # Constraints: Capital requirement
        capital_coefficients = [
            customer['Initial Loan ($)'] * self.capital_requirement_ratio 
            for _, customer in self.data.iterrows()
        ]
        
        A_ub = [capital_coefficients]  # Capital constraint
        b_ub = [500000]  # Maximum capital available
        
        # Bounds: 0 <= x_i <= 1 (fraction of increase granted)
        bounds = [(0, 1) for _ in range(n_customers)]
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        return result
    
    def reinforcement_learning_policy(self, n_episodes=1000):
        """Implement Q-learning for optimal policy"""
        # Simplified Q-learning implementation
        n_states = len(self.risk_states)
        n_actions = 2
        
        # Initialize Q-table
        Q = np.zeros((n_states, n_actions))
        
        # Learning parameters
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.1  # Exploration rate
        
        for episode in range(n_episodes):
            # Start with random state
            state_idx = np.random.randint(0, n_states)
            total_reward = 0
            
            for step in range(10):  # 10 steps per episode
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(0, n_actions)
                else:
                    action = np.argmax(Q[state_idx])
                
                # Simulate reward (simplified)
                state = self.risk_states[state_idx]
                reward = self.calculate_expected_profit(1000, state, action)
                
                # Next state (simplified transition)
                next_state_idx = np.random.randint(0, n_states)
                
                # Update Q-value
                Q[state_idx, action] = (1 - alpha) * Q[state_idx, action] + \
                                      alpha * (reward + gamma * np.max(Q[next_state_idx]))
                
                state_idx = next_state_idx
                total_reward += reward
        
        return Q
    
    def demand_forecasting(self, economic_factors=None):
        """Forecast customer uptake of loan increases"""
        if economic_factors is None:
            economic_factors = {
                'inflation_rate': 0.03,
                'unemployment_rate': 0.05,
                'interest_rate': 0.07
            }
        
        # Feature engineering for demand prediction
        self.data['Risk_State'] = self.data['On-time Payments (%)'].apply(self.classify_risk_state)
        self.data['Days_Since_Last_Category'] = pd.cut(
            self.data['Days Since Last Loan'], 
            bins=[0, 30, 60, 90, 180, 365, float('inf')]
        )
        
        # Simplified demand model
        base_acceptance_rate = 0.6
        economic_impact = 1 - (economic_factors['unemployment_rate'] * 0.5 + 
                              economic_factors['inflation_rate'] * 0.3)
        
        risk_adjustment = {
            'Prime': 1.2,
            'Near-Prime': 1.0,
            'Subprime': 0.7
        }
        
        self.data['Predicted_Uptake'] = self.data['Risk_State'].map(risk_adjustment) * \
                                      base_acceptance_rate * economic_impact
        
        return self.data['Predicted_Uptake'].mean()
    
    def optimize_strategy(self):
        """Main optimization method combining all approaches"""
        print("Starting loan limit optimization...")
        
        # 1. Demand forecasting
        avg_uptake = self.demand_forecasting()
        print(f"Predicted average uptake rate: {avg_uptake:.2%}")
        
        # 2. Monte Carlo simulation
        print("Running Monte Carlo simulation...")
        mc_results = self.monte_carlo_simulation(n_simulations=500)
        print(f"Expected profit distribution: Mean=${mc_results.mean():.2f}, Std=${mc_results.std():.2f}")
        
        # 3. Linear programming optimization
        print("Solving linear programming problem...")
        lp_result = self.linear_programming_optimization()
        print(f"LP optimization status: {lp_result.status}")
        if lp_result.success:
            print(f"Optimal objective value: ${-lp_result.fun:.2f}")
        else:
            print("LP optimization failed to find optimal solution")
        
        # 4. Reinforcement learning policy
        print("Training reinforcement learning policy...")
        q_table = self.reinforcement_learning_policy(n_episodes=500)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(q_table)
        
        return {
            'monte_carlo_results': mc_results,
            'lp_solution': lp_result,
            'q_table': q_table,
            'recommendations': recommendations,
            'demand_forecast': avg_uptake,
            'optimizer': self  # Include optimizer instance for analysis
        }
    
    def generate_recommendations(self, q_table):
        """Generate actionable recommendations based on optimization results"""
        recommendations = []
        
        for _, customer in self.data.iterrows():
            risk_state = self.classify_risk_state(customer['On-time Payments (%)'])
            state_idx = self.risk_states.index(risk_state)
            
            # Get optimal action from Q-table
            optimal_action = np.argmax(q_table[state_idx])
            
            recommendation = {
                'Customer_ID': customer['Customer ID'],
                'Risk_State': risk_state,
                'Recommended_Action': 'GRANT' if optimal_action == 1 else 'DENY',
                'Expected_Profit': self.calculate_expected_profit(
                    customer['Initial Loan ($)'], risk_state, optimal_action
                ),
                'Days_Since_Last_Loan': customer['Days Since Last Loan'],
                'On_Time_Payments': customer['On-time Payments (%)']
            }
            recommendations.append(recommendation)
        
        return pd.DataFrame(recommendations)

    def analyze_results(self, results):
        """Analyze and visualize optimization results"""
        
        # 1. Profit distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(results['monte_carlo_results'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Monte Carlo Simulation: Profit Distribution')
        plt.xlabel('Profit ($)')
        plt.ylabel('Frequency')
        
        # 2. Risk state distribution
        plt.subplot(2, 2, 2)
        risk_states = self.data['On-time Payments (%)'].apply(self.classify_risk_state)
        risk_counts = risk_states.value_counts()
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        plt.title('Customer Risk State Distribution')
        
        # 3. Action recommendations by risk state
        plt.subplot(2, 2, 3)
        recommendations = results['recommendations']
        action_by_risk = pd.crosstab(recommendations['Risk_State'], recommendations['Recommended_Action'])
        action_by_risk.plot(kind='bar', ax=plt.gca())
        plt.title('Recommended Actions by Risk State')
        plt.xlabel('Risk State')
        plt.ylabel('Count')
        plt.legend(title='Action')
        
        # 4. Expected profit by risk state
        plt.subplot(2, 2, 4)
        profit_by_risk = recommendations.groupby('Risk_State')['Expected_Profit'].mean()
        plt.bar(profit_by_risk.index, profit_by_risk.values)
        plt.title('Average Expected Profit by Risk State')
        plt.xlabel('Risk State')
        plt.ylabel('Expected Profit ($)')
        
        plt.tight_layout()
        plt.show()
        
        return risk_counts, profit_by_risk

    def generate_strategic_insights(self, results):
        """Generate strategic business insights"""
        
        recommendations = results['recommendations']
        
        print("\n" + "="*60)
        print("STRATEGIC INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # 1. Overall strategy
        total_customers = len(recommendations)
        grant_recommendations = len(recommendations[recommendations['Recommended_Action'] == 'GRANT'])
        grant_rate = grant_recommendations / total_customers
        
        print(f"\n1. OVERALL STRATEGY:")
        print(f"   - Recommended grant rate: {grant_rate:.1%}")
        print(f"   - Total customers analyzed: {total_customers:,}")
        print(f"   - Customers recommended for increase: {grant_recommendations:,}")
        
        # 2. Risk-based insights
        print(f"\n2. RISK-BASED INSIGHTS:")
        for risk_state in ['Prime', 'Near-Prime', 'Subprime']:
            risk_customers = recommendations[recommendations['Risk_State'] == risk_state]
            if len(risk_customers) > 0:
                risk_grant_rate = len(risk_customers[risk_customers['Recommended_Action'] == 'GRANT']) / len(risk_customers)
                avg_profit = risk_customers['Expected_Profit'].mean()
                
                print(f"   - {risk_state}:")
                print(f"     * Grant rate: {risk_grant_rate:.1%}")
                print(f"     * Avg expected profit: ${avg_profit:.2f}")
            else:
                print(f"   - {risk_state}: No customers in this category")
        
        # 3. Economic sensitivity
        print(f"\n3. ECONOMIC SENSITIVITY ANALYSIS:")
        economic_scenarios = [
            {'inflation_rate': 0.02, 'unemployment_rate': 0.03, 'interest_rate': 0.05},  # Boom
            {'inflation_rate': 0.03, 'unemployment_rate': 0.05, 'interest_rate': 0.07},  # Normal
            {'inflation_rate': 0.06, 'unemployment_rate': 0.08, 'interest_rate': 0.10},  # Recession
        ]
        
        scenario_names = ['Economic Boom', 'Normal Conditions', 'Recession']
        
        for scenario, name in zip(economic_scenarios, scenario_names):
            uptake_rate = self.demand_forecasting(scenario)
            print(f"   - {name}: Predicted uptake rate = {uptake_rate:.1%}")
        
        # 4. Implementation recommendations
        print(f"\n4. IMPLEMENTATION RECOMMENDATIONS:")
        print("   - Implement dynamic risk-based pricing for limit increases")
        print("   - Use reinforcement learning for real-time policy adaptation")
        print("   - Monitor economic indicators for strategy adjustments")
        print("   - Implement behavioral nudges for high-risk customers")
        print("   - Establish regular model retraining schedule (quarterly)")
        
        # 5. Key performance metrics
        total_expected_profit = recommendations['Expected_Profit'].sum()
        avg_profit_per_customer = recommendations['Expected_Profit'].mean()
        
        print(f"\n5. KEY PERFORMANCE METRICS:")
        print(f"   - Total expected profit: ${total_expected_profit:,.2f}")
        print(f"   - Average profit per customer: ${avg_profit_per_customer:.2f}")
        print(f"   - Monte Carlo expected profit: ${results['monte_carlo_results'].mean():.2f}")

# Simplified main function
def main():
    # Since we have the data in the current environment, let's use it directly
    # Convert the file content to DataFrame
    print("Processing dataset...")
    
    # Extract data from the provided file content
    data_lines = []
    for line in file_content.split('\n'):
        if '|' in line and not line.startswith('>') and not line.startswith('| A'):
            # Clean and parse the line
            clean_line = line.strip().strip('|')
            if clean_line:
                data_lines.append([x.strip() for x in clean_line.split('|')])
    
    # Create DataFrame
    if data_lines:
        columns = ['Customer ID', 'Initial Loan ($)', 'Days Since Last Loan', 
                  'On-time Payments (%)', 'No. of Increases in 2023', 'Total Profit Contribution ($)']
        df = pd.DataFrame(data_lines[1:], columns=columns)
        
        # Convert data types
        df['Customer ID'] = pd.to_numeric(df['Customer ID'], errors='coerce')
        df['Initial Loan ($)'] = pd.to_numeric(df['Initial Loan ($)'], errors='coerce')
        df['Days Since Last Loan'] = pd.to_numeric(df['Days Since Last Loan'], errors='coerce')
        df['On-time Payments (%)'] = pd.to_numeric(df['On-time Payments (%)'], errors='coerce')
        df['No. of Increases in 2023'] = pd.to_numeric(df['No. of Increases in 2023'], errors='coerce')
        df['Total Profit Contribution ($)'] = pd.to_numeric(df['Total Profit Contribution ($)'], errors='coerce')
        
        # Drop any rows with missing values
        df = df.dropna()
        
        print(f"Dataset loaded successfully: {len(df)} records")
        
        # Initialize and run optimizer
        optimizer = LoanLimitOptimizer(df)
        results = optimizer.optimize_strategy()
        
        # Analyze results using optimizer's methods
        risk_counts, profit_by_risk = optimizer.analyze_results(results)
        optimizer.generate_strategic_insights(results)
        
        return optimizer, results
    else:
        print("Error: Could not parse data from file content")
        return None, None

# Alternative: If running in notebook with data already loaded
def run_analysis_with_data(df):
    """Run analysis with provided DataFrame"""
    print(f"Starting analysis with {len(df)} records...")
    
    # Initialize and run optimizer
    optimizer = LoanLimitOptimizer(df)
    results = optimizer.optimize_strategy()
    
    # Analyze results
    risk_counts, profit_by_risk = optimizer.analyze_results(results)
    optimizer.generate_strategic_insights(results)
    
    return optimizer, results

# Run the analysis
if __name__ == "__main__":
    # For notebook environment, you can call:
    # optimizer, results = run_analysis_with_data(your_dataframe)
    
    # For standalone execution
    optimizer, results = main()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class SimpleLoanOptimizer:
    def __init__(self):
        self.model = None
        self.risk_threshold = 0.7  # Probability threshold for granting increases
        
    def prepare_features(self, df):
        """Create features for the ML model"""
        # Basic customer features
        df['risk_category'] = df['On-time Payments (%)'].apply(
            lambda x: 'prime' if x >= 95 else 'near_prime' if x >= 85 else 'subprime'
        )
        
        # Feature engineering
        df['days_since_last_loan'] = df['Days Since Last Loan']
        df['loan_amount'] = df['Initial Loan ($)']
        df['previous_increases'] = df['No. of Increases in 2023']
        df['payment_performance'] = df['On-time Payments (%)'] / 100
        
        # Eligibility: 60+ days since last loan and <6 increases
        df['is_eligible'] = ((df['days_since_last_loan'] >= 60) & 
                           (df['previous_increases'] < 6)).astype(int)
        
        # Create feature matrix
        features = ['loan_amount', 'days_since_last_loan', 'previous_increases', 
                   'payment_performance', 'is_eligible']
        
        return df[features]
    
    def calculate_expected_profit(self, row, approval_prob):
        """Calculate expected profit for granting increase"""
        base_profit = 40
        risk_multiplier = 1 - (100 - row['On-time Payments (%)']) / 100
        
        # Default probability based on payment history
        default_prob = max(0, (100 - row['On-time Payments (%)']) / 500)  # 0-20% default risk
        
        expected_profit = (base_profit * risk_multiplier * (1 - default_prob) * approval_prob -
                          row['Initial Loan ($)'] * default_prob * 0.1)  # 10% loss given default
        
        return expected_profit
    
    def train_model(self, df):
        """Train a simple Random Forest classifier"""
        print("Training ML model...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = (df['Total Profit Contribution ($)'] > 0).astype(int)  # Positive profit as target
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def predict_approval_probability(self, df):
        """Predict probability of profitable approval"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X = self.prepare_features(df)
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of positive profit
        
        return probabilities
    
    def optimize_decisions(self, df, capital_budget=1000000):
        """Optimize loan increase decisions"""
        print("\nOptimizing loan limit increases...")
        
        # Predict approval probabilities
        approval_probs = self.predict_approval_probability(df)
        
        # Calculate expected profits
        expected_profits = []
        for i, row in df.iterrows():
            profit = self.calculate_expected_profit(row, approval_probs[i])
            expected_profits.append(profit)
        
        df['expected_profit'] = expected_profits
        df['approval_probability'] = approval_probs
        
        # Filter eligible customers
        eligible_df = df[df['is_eligible'] == 1].copy()
        
        # Sort by expected profit
        eligible_df = eligible_df.sort_values('expected_profit', ascending=False)
        
        # Apply capital constraints (simplified)
        capital_used = 0
        decisions = []
        
        for _, customer in eligible_df.iterrows():
            capital_needed = customer['Initial Loan ($)'] * 0.08  # 8% capital requirement
            
            if (capital_used + capital_needed <= capital_budget and 
                customer['approval_probability'] >= self.risk_threshold and
                customer['expected_profit'] > 0):
                
                decisions.append({
                    'Customer_ID': customer['Customer ID'],
                    'Approve': True,
                    'Expected_Profit': customer['expected_profit'],
                    'Risk_Category': 'prime' if customer['On-time Payments (%)'] >= 95 else 
                                   'near_prime' if customer['On-time Payments (%)'] >= 85 else 'subprime',
                    'Capital_Required': capital_needed
                })
                capital_used += capital_needed
            else:
                decisions.append({
                    'Customer_ID': customer['Customer ID'],
                    'Approve': False,
                    'Expected_Profit': customer['expected_profit'],
                    'Risk_Category': 'prime' if customer['On-time Payments (%)'] >= 95 else 
                                   'near_prime' if customer['On-time Payments (%)'] >= 85 else 'subprime',
                    'Capital_Required': capital_needed
                })
        
        decisions_df = pd.DataFrame(decisions)
        
        # Summary statistics
        total_approved = decisions_df['Approve'].sum()
        total_expected_profit = decisions_df[decisions_df['Approve']]['Expected_Profit'].sum()
        capital_utilization = capital_used / capital_budget
        
        print(f"\nOptimization Results:")
        print(f"Customers approved: {total_approved}/{len(eligible_df)}")
        print(f"Total expected profit: ${total_expected_profit:,.2f}")
        print(f"Capital utilization: {capital_utilization:.1%}")
        print(f"Average profit per approved customer: ${total_expected_profit/max(1, total_approved):.2f}")
        
        return decisions_df

# Monte Carlo Simulation for Risk Assessment
class RiskSimulator:
    def __init__(self):
        self.default_rates = {
            'prime': 0.01,
            'near_prime': 0.05, 
            'subprime': 0.15
        }
    
    def simulate_loan_performance(self, df, n_simulations=1000):
        """Simulate loan performance under different economic scenarios"""
        print("\nRunning Monte Carlo Simulation...")
        
        results = []
        
        for sim in range(n_simulations):
            total_profit = 0
            total_defaults = 0
            
            for _, customer in df.iterrows():
                # Determine risk category
                if customer['On-time Payments (%)'] >= 95:
                    risk_cat = 'prime'
                elif customer['On-time Payments (%)'] >= 85:
                    risk_cat = 'near_prime'
                else:
                    risk_cat = 'subprime'
                
                # Simulate default
                default_prob = self.default_rates[risk_cat]
                defaults = np.random.random() < default_prob
                
                if defaults:
                    total_defaults += 1
                    profit = -customer['Initial Loan ($)'] * 0.5  # 50% loss given default
                else:
                    profit = 40  # Base profit from increase
                
                total_profit += profit
            
            results.append({
                'simulation': sim,
                'total_profit': total_profit,
                'default_rate': total_defaults / len(df)
            })
        
        results_df = pd.DataFrame(results)
        
        print(f"Average profit across {n_simulations} simulations: ${results_df['total_profit'].mean():,.2f}")
        print(f"Average default rate: {results_df['default_rate'].mean():.2%}")
        print(f"Profit standard deviation: ${results_df['total_profit'].std():,.2f}")
        
        return results_df

# Economic Factor Analysis
class EconomicAnalyzer:
    def __init__(self):
        self.economic_factors = {
            'recession': {'inflation': 0.06, 'unemployment': 0.08, 'growth': -0.02},
            'normal': {'inflation': 0.03, 'unemployment': 0.05, 'growth': 0.02},
            'boom': {'inflation': 0.02, 'unemployment': 0.03, 'growth': 0.04}
        }
    
    def analyze_economic_impact(self, df):
        """Analyze how economic conditions affect approval strategy"""
        print("\nEconomic Impact Analysis:")
        
        for scenario, factors in self.economic_factors.items():
            # Adjust risk thresholds based on economic conditions
            if scenario == 'recession':
                risk_adjustment = 1.2  # More conservative
                uptake_reduction = 0.8  # 20% lower uptake
            elif scenario == 'boom':
                risk_adjustment = 0.8   # Less conservative  
                uptake_reduction = 1.2  # 20% higher uptake
            else:
                risk_adjustment = 1.0
                uptake_reduction = 1.0
            
            # Calculate adjusted metrics
            adjusted_profits = df['expected_profit'] * uptake_reduction * risk_adjustment
            viable_customers = len(adjusted_profits[adjusted_profits > 0])
            
            print(f"{scenario.upper()} scenario:")
            print(f"  - Viable customers: {viable_customers}")
            print(f"  - Average adjusted profit: ${adjusted_profits.mean():.2f}")
            print(f"  - Economic factors: {factors}")

# Main execution function
def main():
    # Load your data
    # df = pd.read_csv('loan_limit_increases.csv')
    
    # For demonstration, create sample data
    print("Creating sample data...")
    np.random.seed(42)
    n_customers = 30000
    
    sample_data = {
        'Customer ID': range(1, n_customers + 1),
        'Initial Loan ($)': np.random.normal(3000, 1000, n_customers).clip(500, 10000),
        'Days Since Last Loan': np.random.exponential(90, n_customers).clip(1, 365),
        'On-time Payments (%)': np.random.normal(90, 10, n_customers).clip(60, 100),
        'No. of Increases in 2023': np.random.poisson(2, n_customers).clip(0, 6),
        'Total Profit Contribution ($)': np.random.normal(50, 30, n_customers)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize and run the optimizer
    optimizer = SimpleLoanOptimizer()
    
    # Train ML model
    optimizer.train_model(df)
    
    # Optimize decisions
    decisions = optimizer.optimize_decisions(df)
    
    # Risk simulation
    simulator = RiskSimulator()
    simulation_results = simulator.simulate_loan_performance(df[decisions['Approve']])
    
    # Economic analysis
    economic_analyzer = EconomicAnalyzer()
    economic_analyzer.analyze_economic_impact(df)
    
    # Generate insights
    print("\n" + "="*50)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*50)
    
    # Risk-based approval rates
    risk_approvals = decisions.groupby('Risk_Category')['Approve'].mean()
    print("\n1. Approval Rates by Risk Category:")
    for risk, rate in risk_approvals.items():
        print(f"   - {risk}: {rate:.1%}")
    
    # Profitability by segment
    profit_by_risk = decisions[decisions['Approve']].groupby('Risk_Category')['Expected_Profit'].mean()
    print("\n2. Average Profit by Risk Category:")
    for risk, profit in profit_by_risk.items():
        print(f"   - {risk}: ${profit:.2f}")
    
    # Strategic recommendations
    print("\n3. Strategic Recommendations:")
    print("   - Focus on prime and near-prime customers for highest ROI")
    print("   - Implement dynamic pricing based on risk categories")
    print("   - Use economic indicators to adjust risk thresholds")
    print("   - Monitor portfolio concentration across risk segments")
    print("   - Regular model retraining with latest performance data")

# Mathematical Formulation
"""
MATHEMATICAL FORMULATION:

Objective: Maximize total expected profit
Maximize: Σ [P(approve_i) * (Profit_i * (1 - PD_i) - LGD_i * PD_i)]

Where:
- P(approve_i) = Probability of approving customer i
- Profit_i = $40 base profit
- PD_i = Probability of default for customer i
- LGD_i = Loss given default (50% of loan amount)

Constraints:
1. Capital: Σ [Capital_i * Approve_i] ≤ Budget
2. Frequency: Increases_i ≤ 6 per year
3. Eligibility: Days_since_last_loan_i ≥ 60
4. Risk: PD_i ≤ Maximum acceptable risk

Decision Variables:
Approve_i ∈ {0, 1} for each customer i
"""

if __name__ == "__main__":
    main()
df
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class SimpleLoanOptimizer:
    def __init__(self):
        self.model = None
        self.risk_threshold = 0.7  # Probability threshold for granting increases
        
    def prepare_features(self, df):
        """Create features for the ML model"""
        # Basic customer features
        df['risk_category'] = df['On-time Payments (%)'].apply(
            lambda x: 'prime' if x >= 95 else 'near_prime' if x >= 85 else 'subprime'
        )
        
        # Feature engineering
        df['days_since_last_loan'] = df['Days Since Last Loan']
        df['loan_amount'] = df['Initial Loan ($)']
        df['previous_increases'] = df['No. of Increases in 2023']
        df['payment_performance'] = df['On-time Payments (%)'] / 100
        
        # Eligibility: 60+ days since last loan and <6 increases
        df['is_eligible'] = ((df['days_since_last_loan'] >= 60) & 
                           (df['previous_increases'] < 6)).astype(int)
        
        # Create feature matrix
        features = ['loan_amount', 'days_since_last_loan', 'previous_increases', 
                   'payment_performance', 'is_eligible']
        
        return df[features]
    
    def calculate_expected_profit(self, row, approval_prob):
        """Calculate expected profit for granting increase"""
        base_profit = 40
        risk_multiplier = 1 - (100 - row['On-time Payments (%)']) / 100
        
        # Default probability based on payment history
        default_prob = max(0, (100 - row['On-time Payments (%)']) / 500)  # 0-20% default risk
        
        expected_profit = (base_profit * risk_multiplier * (1 - default_prob) * approval_prob -
                          row['Initial Loan ($)'] * default_prob * 0.1)  # 10% loss given default
        
        return expected_profit
    
    def train_model(self, df):
        """Train a simple Random Forest classifier"""
        print("Training ML model...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = (df['Total Profit Contribution ($)'] > 0).astype(int)  # Positive profit as target
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def predict_approval_probability(self, df):
        """Predict probability of profitable approval"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X = self.prepare_features(df)
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of positive profit
        
        return probabilities
    
    def optimize_decisions(self, df, capital_budget=1000000):
        """Optimize loan increase decisions"""
        print("\nOptimizing loan limit increases...")
        
        # Predict approval probabilities
        approval_probs = self.predict_approval_probability(df)
        
        # Calculate expected profits
        expected_profits = []
        for i, row in df.iterrows():
            profit = self.calculate_expected_profit(row, approval_probs[i])
            expected_profits.append(profit)
        
        df['expected_profit'] = expected_profits
        df['approval_probability'] = approval_probs
        
        # Filter eligible customers
        eligible_df = df[df['is_eligible'] == 1].copy()
        
        # Sort by expected profit
        eligible_df = eligible_df.sort_values('expected_profit', ascending=False)
        
        # Apply capital constraints (simplified)
        capital_used = 0
        decisions = []
        
        for _, customer in eligible_df.iterrows():
            capital_needed = customer['Initial Loan ($)'] * 0.08  # 8% capital requirement
            
            if (capital_used + capital_needed <= capital_budget and 
                customer['approval_probability'] >= self.risk_threshold and
                customer['expected_profit'] > 0):
                
                decisions.append({
                    'Customer_ID': customer['Customer ID'],
                    'Approve': True,
                    'Expected_Profit': customer['expected_profit'],
                    'Risk_Category': 'prime' if customer['On-time Payments (%)'] >= 95 else 
                                   'near_prime' if customer['On-time Payments (%)'] >= 85 else 'subprime',
                    'Capital_Required': capital_needed
                })
                capital_used += capital_needed
            else:
                decisions.append({
                    'Customer_ID': customer['Customer ID'],
                    'Approve': False,
                    'Expected_Profit': customer['expected_profit'],
                    'Risk_Category': 'prime' if customer['On-time Payments (%)'] >= 95 else 
                                   'near_prime' if customer['On-time Payments (%)'] >= 85 else 'subprime',
                    'Capital_Required': capital_needed
                })
        
        decisions_df = pd.DataFrame(decisions)
        
        # Summary statistics
        total_approved = decisions_df['Approve'].sum()
        total_expected_profit = decisions_df[decisions_df['Approve']]['Expected_Profit'].sum()
        capital_utilization = capital_used / capital_budget
        
        print(f"\nOptimization Results:")
        print(f"Customers approved: {total_approved}/{len(eligible_df)}")
        print(f"Total expected profit: ${total_expected_profit:,.2f}")
        print(f"Capital utilization: {capital_utilization:.1%}")
        print(f"Average profit per approved customer: ${total_expected_profit/max(1, total_approved):.2f}")
        
        return decisions_df, df  # Return both decisions and updated df

# Monte Carlo Simulation for Risk Assessment
class RiskSimulator:
    def __init__(self):
        self.default_rates = {
            'prime': 0.01,
            'near_prime': 0.05, 
            'subprime': 0.15
        }
    
    def simulate_loan_performance(self, approved_customers_df):
        """Simulate loan performance for approved customers"""
        print("\nRunning Monte Carlo Simulation...")
        
        if len(approved_customers_df) == 0:
            print("No approved customers to simulate")
            return pd.DataFrame()
            
        n_simulations = 1000
        results = []
        
        for sim in range(n_simulations):
            total_profit = 0
            total_defaults = 0
            
            for _, customer in approved_customers_df.iterrows():
                # Determine risk category
                if customer['On-time Payments (%)'] >= 95:
                    risk_cat = 'prime'
                elif customer['On-time Payments (%)'] >= 85:
                    risk_cat = 'near_prime'
                else:
                    risk_cat = 'subprime'
                
                # Simulate default
                default_prob = self.default_rates[risk_cat]
                defaults = np.random.random() < default_prob
                
                if defaults:
                    total_defaults += 1
                    profit = -customer['Initial Loan ($)'] * 0.5  # 50% loss given default
                else:
                    profit = 40  # Base profit from increase
                
                total_profit += profit
            
            results.append({
                'simulation': sim,
                'total_profit': total_profit,
                'default_rate': total_defaults / len(approved_customers_df)
            })
        
        results_df = pd.DataFrame(results)
        
        print(f"Average profit across {n_simulations} simulations: ${results_df['total_profit'].mean():,.2f}")
        print(f"Average default rate: {results_df['default_rate'].mean():.2%}")
        print(f"Profit standard deviation: ${results_df['total_profit'].std():,.2f}")
        
        return results_df

# Economic Factor Analysis
class EconomicAnalyzer:
    def __init__(self):
        self.economic_factors = {
            'recession': {'inflation': 0.06, 'unemployment': 0.08, 'growth': -0.02},
            'normal': {'inflation': 0.03, 'unemployment': 0.05, 'growth': 0.02},
            'boom': {'inflation': 0.02, 'unemployment': 0.03, 'growth': 0.04}
        }
    
    def analyze_economic_impact(self, df):
        """Analyze how economic conditions affect approval strategy"""
        print("\nEconomic Impact Analysis:")
        
        for scenario, factors in self.economic_factors.items():
            # Adjust risk thresholds based on economic conditions
            if scenario == 'recession':
                risk_adjustment = 1.2  # More conservative
                uptake_reduction = 0.8  # 20% lower uptake
            elif scenario == 'boom':
                risk_adjustment = 0.8   # Less conservative  
                uptake_reduction = 1.2  # 20% higher uptake
            else:
                risk_adjustment = 1.0
                uptake_reduction = 1.0
            
            # Calculate adjusted metrics
            if 'expected_profit' in df.columns:
                adjusted_profits = df['expected_profit'] * uptake_reduction * risk_adjustment
                viable_customers = len(adjusted_profits[adjusted_profits > 0])
                
                print(f"{scenario.upper()} scenario:")
                print(f"  - Viable customers: {viable_customers}")
                print(f"  - Average adjusted profit: ${adjusted_profits.mean():.2f}")
                print(f"  - Economic factors: {factors}")

# Main execution function
def main():
    # Load your data
    # df = pd.read_csv('loan_limit_increases.csv')
    
    # For demonstration, create sample data
    print("Creating sample data...")
    np.random.seed(42)
    n_customers = 3000  # Reduced for faster execution
    
    sample_data = {
        'Customer ID': range(1, n_customers + 1),
        'Initial Loan ($)': np.random.normal(3000, 1000, n_customers).clip(500, 10000),
        'Days Since Last Loan': np.random.exponential(90, n_customers).clip(1, 365),
        'On-time Payments (%)': np.random.normal(90, 10, n_customers).clip(60, 100),
        'No. of Increases in 2023': np.random.poisson(2, n_customers).clip(0, 6),
        'Total Profit Contribution ($)': np.random.normal(50, 30, n_customers)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize and run the optimizer
    optimizer = SimpleLoanOptimizer()
    
    # Train ML model
    optimizer.train_model(df)
    
    # Optimize decisions - now returns both decisions and updated df
    decisions, updated_df = optimizer.optimize_decisions(df)
    
    # Get approved customer IDs
    approved_customer_ids = decisions[decisions['Approve']]['Customer_ID'].values
    
    # Get approved customers from original dataframe
    approved_customers = df[df['Customer ID'].isin(approved_customer_ids)]
    
    # Risk simulation with approved customers
    simulator = RiskSimulator()
    simulation_results = simulator.simulate_loan_performance(approved_customers)
    
    # Economic analysis
    economic_analyzer = EconomicAnalyzer()
    economic_analyzer.analyze_economic_impact(updated_df)
    
    # Generate insights
    print("\n" + "="*50)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*50)
    
    # Risk-based approval rates
    risk_approvals = decisions.groupby('Risk_Category')['Approve'].mean()
    print("\n1. Approval Rates by Risk Category:")
    for risk, rate in risk_approvals.items():
        print(f"   - {risk}: {rate:.1%}")
    
    # Profitability by segment
    if len(decisions[decisions['Approve']]) > 0:
        profit_by_risk = decisions[decisions['Approve']].groupby('Risk_Category')['Expected_Profit'].mean()
        print("\n2. Average Profit by Risk Category:")
        for risk, profit in profit_by_risk.items():
            print(f"   - {risk}: ${profit:.2f}")
    else:
        print("\n2. No approved customers for profit analysis")
    
    # Strategic recommendations
    print("\n3. Strategic Recommendations:")
    print("   - Focus on prime and near-prime customers for highest ROI")
    print("   - Implement dynamic pricing based on risk categories")
    print("   - Use economic indicators to adjust risk thresholds")
    print("   - Monitor portfolio concentration across risk segments")
    print("   - Regular model retraining with latest performance data")
    
    return decisions, updated_df, simulation_results

# Mathematical Formulation
"""
MATHEMATICAL FORMULATION:

Objective: Maximize total expected profit
Maximize: Σ [P(approve_i) * (Profit_i * (1 - PD_i) - LGD_i * PD_i)]

Where:
- P(approve_i) = Probability of approving customer i
- Profit_i = $40 base profit
- PD_i = Probability of default for customer i
- LGD_i = Loss given default (50% of loan amount)

Constraints:
1. Capital: Σ [Capital_i * Approve_i] ≤ Budget
2. Frequency: Increases_i ≤ 6 per year
3. Eligibility: Days_since_last_loan_i ≥ 60
4. Risk: PD_i ≤ Maximum acceptable risk

Decision Variables:
Approve_i ∈ {0, 1} for each customer i
"""

if __name__ == "__main__":
    decisions, updated_df, simulation_results = main()
df
df
updated_df
%history -f loan_limit_increases.py
updated_df.to_csv("analyzed loan_limit_increase",index=false)
updated_df.to_csv("analyzed loan_limit_increase",index=False)
updated_df.to_csv("C:\Users\Administrator\Downloads\analyzed loan_limit_increase.xlsx"index=False)
updated_df.to_csv(r"C:\Users\Administrator\Downloads\analyzed loan_limit_increase.xlsx"index=False)
updated_df.to_csv(r"C:\Users\Administrator\Downloads\analyzed loan_limit_increase.xlsx",index=False)
updated_df.to_excel(r"C:\Users\Administrator\Downloads\analyzed loan_limit_increase.xlsx",index=False)
% history -f loan_limit_increases.py
%history -f loan_limit_increases.py
