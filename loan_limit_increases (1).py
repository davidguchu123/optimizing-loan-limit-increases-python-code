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
    
updated_df.to_excel(r"C:\Users\Administrator\Downloads\analyzed loan_limit_increase.xlsx",index=False)
