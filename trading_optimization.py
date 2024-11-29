import streamlit as st
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import plotly.express as px

class TradingAccountOptimizer:
    def __init__(self):
        self.num_accounts = 0
        self.initial_capitals = []
        self.final_amounts = []
        self.profit_secure_percentage = 0
        self.total_redistribution_amount = 0
        self.additional_funds = 0

    def calculate_account_metrics(self):
        """Calculate performance metrics for each account"""
        metrics = {
            'initial_capitals': self.initial_capitals,
            'final_amounts': self.final_amounts,
            'profits': [],
            'roi': [],
            'risk_adjusted_returns': [],
            'total_pnl': round(sum(self.final_amounts) - sum(self.initial_capitals), 2)
        }

        for init, final in zip(self.initial_capitals, self.final_amounts):
            profit = round(final - init, 2)
            roi = round((profit / init) * 100, 2) if init != 0 else 0

            metrics['profits'].append(profit)
            metrics['roi'].append(roi)

            if metrics['total_pnl'] >= 0:
                risk_adjusted = roi / (abs(roi) + 1)
            else:
                risk_adjusted = 1 / (abs(roi) + 1)

            metrics['risk_adjusted_returns'].append(round(risk_adjusted, 4))

        return metrics

    def calculate_redistribution_amount(self, metrics):
        """Calculate total amount available for redistribution"""
        if metrics['total_pnl'] > 0:
            total_profit = metrics['total_pnl']
            secure_amount = round((total_profit * self.profit_secure_percentage) / 100, 2)
            redistribution_amount = round(total_profit - secure_amount + sum(self.initial_capitals), 2)
        else:
            redistribution_amount = round(sum(self.final_amounts) + self.additional_funds, 2)

        return max(redistribution_amount, 0)

    def objective_function(self, x, metrics):
        """Objective function to minimize"""
        performance_weights = np.array(metrics['risk_adjusted_returns'])
        performance_weights = (performance_weights - np.min(performance_weights)) + 1
        ideal_distribution = performance_weights / np.sum(performance_weights)

        actual_distribution = x / np.sum(x)
        distribution_penalty = np.sum((actual_distribution - ideal_distribution) ** 2)
        diversity_penalty = np.sum((x / np.sum(x) - 1/len(x)) ** 2)

        if metrics['total_pnl'] >= 0:
            return distribution_penalty + 0.5 * diversity_penalty
        else:
            return distribution_penalty + diversity_penalty

    def optimize_distribution(self):
        """Optimize the distribution of money across accounts"""
        metrics = self.calculate_account_metrics()
        self.total_redistribution_amount = self.calculate_redistribution_amount(metrics)

        x0 = np.ones(self.num_accounts) / self.num_accounts
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if metrics['total_pnl'] >= 0:
            bounds = [(0.05, 0.5) for _ in range(self.num_accounts)]
        else:
            bounds = [(0.1, 0.4) for _ in range(self.num_accounts)]

        result = minimize(
            self.objective_function,
            x0,
            args=(metrics,),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )

        final_distribution = [round(amt, 2) for amt in (result.x * self.total_redistribution_amount)]
        rounding_difference = round(self.total_redistribution_amount - sum(final_distribution), 2)
        if rounding_difference != 0:
            max_index = final_distribution.index(max(final_distribution))
            final_distribution[max_index] = round(final_distribution[max_index] + rounding_difference, 2)

        return final_distribution

    def streamlit_inputs(self):
        """Get all required inputs using Streamlit widgets"""
        st.title("Trading Account Optimizer")
        
        self.num_accounts = st.number_input("Number of trading accounts", 
                                          min_value=1, max_value=10, value=2)
        
        col1, col2 = st.columns(2)
        
        self.initial_capitals = []
        self.final_amounts = []
        
        with col1:
            st.subheader("Initial Capitals")
            for i in range(self.num_accounts):
                capital = st.number_input(
                    f"Initial capital for Account {i+1}", 
                    min_value=0.0, 
                    value=1000.0,
                    key=f"init_{i}"
                )
                self.initial_capitals.append(round(capital, 2))
                
        with col2:
            st.subheader("Final Amounts")
            for i in range(self.num_accounts):
                amount = st.number_input(
                    f"Final amount for Account {i+1}", 
                    min_value=0.0,
                    value=1100.0,
                    key=f"final_{i}"
                )
                self.final_amounts.append(round(amount, 2))

        total_initial = sum(self.initial_capitals)
        total_final = sum(self.final_amounts)
        total_pnl = round(total_final - total_initial, 2)

        if total_pnl > 0:
            st.success(f"Congratulations! You have a total profit of ${total_pnl:,.2f}")
            self.profit_secure_percentage = st.slider(
                "Percentage of profit to secure", 
                0, 100, 50
            )
        else:
            st.error(f"Your portfolio has a loss of ${abs(total_pnl):,.2f}")
            choice = st.radio(
                "Choose an option:",
                ["Add additional funds", "Redistribute existing funds"]
            )
            
            if choice == "Add additional funds":
                self.additional_funds = st.number_input(
                    "Additional funds to add",
                    min_value=0.0,
                    value=abs(total_pnl)
                )
            else:
                self.additional_funds = 0

    def display_streamlit_results(self, optimized_distribution):
        """Display the optimization results using Streamlit"""
        metrics = self.calculate_account_metrics()
        
        st.header("Optimization Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if metrics['total_pnl'] > 0:
                st.metric("Total Profit", f"${metrics['total_pnl']:,.2f}")
            else:
                st.metric("Total Loss", f"${abs(metrics['total_pnl']):,.2f}")
                
        with col2:
            if metrics['total_pnl'] > 0:
                secured_profit = round((metrics['total_pnl'] * self.profit_secure_percentage / 100), 2)
                st.metric("Secured Profit", f"${secured_profit:,.2f}")
            elif self.additional_funds > 0:
                st.metric("Additional Funds", f"${self.additional_funds:,.2f}")
                
        with col3:
            st.metric("Redistribution Amount", f"${self.total_redistribution_amount:,.2f}")

        results_df = pd.DataFrame({
            'Account': [f"Account {i+1}" for i in range(self.num_accounts)],
            'Initial Capital': self.initial_capitals,
            'Final Amount': self.final_amounts,
            'Profit/Loss': metrics['profits'],
            'ROI (%)': metrics['roi'],
            'New Distribution': optimized_distribution
        })
        
        st.subheader("Detailed Results")
        st.dataframe(results_df.style.format({
            'Initial Capital': '${:,.2f}',
            'Final Amount': '${:,.2f}',
            'Profit/Loss': '${:,.2f}',
            'ROI (%)': '{:.2f}%',
            'New Distribution': '${:,.2f}'
        }))
        
        st.subheader("Visualizations")
        
        chart_data = pd.DataFrame({
            'Account': results_df['Account'],
            'Initial Capital': results_df['Initial Capital'],
            'Final Amount': results_df['Final Amount'],
            'New Distribution': results_df['New Distribution']
        })
        
        chart_data_melted = pd.melt(
            chart_data, 
            id_vars=['Account'], 
            var_name='Stage', 
            value_name='Amount'
        )
        
        fig = px.bar(
            chart_data_melted, 
            x='Account', 
            y='Amount', 
            color='Stage',
            title='Account Distribution Comparison',
            barmode='group'
        )
        st.plotly_chart(fig)

        fig_roi = px.bar(
            results_df, 
            x='Account', 
            y='ROI (%)',
            title='Return on Investment by Account'
        )
        st.plotly_chart(fig_roi)

def main():
    st.set_page_config(page_title="Trading Account Optimizer", layout="wide")
    
    optimizer = TradingAccountOptimizer()
    optimizer.streamlit_inputs()
    
    if st.button("Optimize Distribution"):
        optimized_distribution = optimizer.optimize_distribution()
        optimizer.display_streamlit_results(optimized_distribution)

if __name__ == "__main__":
    main()