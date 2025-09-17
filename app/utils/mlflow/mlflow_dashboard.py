# 2025-01-23T12:15:55_v1.0
"""
MLflow Dashboard for Latent Neurolese Training Run Comparison
Creates interactive dashboard-style visualizations comparing training runs
"""

import mlflow
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MLflowDashboard:
    """Dashboard for comparing MLflow training runs"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5005"):
        """Initialize dashboard with MLflow tracking URI"""
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        
    def get_experiments(self) -> List[Dict]:
        """Get all experiments from MLflow"""
        try:
            # Test connection first
            mlflow.set_tracking_uri(self.tracking_uri)
            experiments = mlflow.search_experiments()
            exp_list = [{"id": exp.experiment_id, "name": exp.name} for exp in experiments]
            logger.info(f"Found {len(exp_list)} experiments: {[e['name'] for e in exp_list]}")
            return exp_list
        except Exception as e:
            logger.error(f"Failed to fetch experiments from {self.tracking_uri}: {e}")
            return []
    
    def get_runs_data(self, experiment_ids: List[str] = None, max_results: int = 50) -> pd.DataFrame:
        """Fetch runs data from MLflow with metrics and parameters"""
        try:
            if experiment_ids:
                runs = []
                for exp_id in experiment_ids:
                    logger.info(f"Fetching runs for experiment ID: {exp_id}")
                    exp_runs = mlflow.search_runs(experiment_ids=[exp_id], max_results=max_results)
                    logger.info(f"Found {len(exp_runs)} runs for experiment {exp_id}")
                    if not exp_runs.empty:
                        runs.append(exp_runs)
                df = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()
            else:
                df = mlflow.search_runs(max_results=max_results)
            
            logger.info(f"Total runs loaded: {len(df) if not df.empty else 0}")
            if df.empty:
                logger.warning("No runs found in selected experiments")
                return pd.DataFrame()
                
            # Clean and prepare data
            df['start_time'] = pd.to_datetime(df['start_time'])
            # Fix datetime handling for duration calculation
            end_time = pd.to_datetime(df['end_time'].fillna(datetime.now()))
            df['duration_minutes'] = (end_time - df['start_time']).dt.total_seconds() / 60
            
            return df
        except Exception as e:
            logger.error(f"Failed to fetch runs data: {e}")
            return pd.DataFrame()
    
    def create_metrics_comparison_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive metrics comparison chart"""
        if df.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", 
                                            x=0.5, y=0.5, showarrow=False)
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Alignment Loss', 'Diversity Loss', 'Total Loss'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Define metrics to plot
        metrics = [
            ('metrics.train_loss', 'Training Loss', 1, 1),
            ('metrics.alignment_loss', 'Alignment Loss', 1, 2),
            ('metrics.diversity_loss', 'Diversity Loss', 2, 1),
            ('metrics.total_loss', 'Total Loss', 2, 2)
        ]
        
        colors = px.colors.qualitative.Set1
        
        for i, (metric_col, title, row, col) in enumerate(metrics):
            if metric_col in df.columns:
                for j, (run_index, run_data) in enumerate(df.iterrows()):
                    # Get actual run_id from the data, not the index
                    actual_run_id = run_data.get('run_id', str(run_index))
                    run_name = run_data.get('tags.mlflow.runName', f'Run {str(actual_run_id)[:8]}')
                    if pd.notna(run_data[metric_col]):
                        fig.add_trace(
                            go.Scatter(
                                x=[0], y=[run_data[metric_col]],
                                mode='markers+text',
                                name=run_name,
                                text=f"{run_data[metric_col]:.4f}",
                                textposition="top center",
                                marker=dict(size=12, color=colors[j % len(colors)]),
                                showlegend=(i == 0)  # Only show legend for first subplot
                            ),
                            row=row, col=col
                        )
        
        fig.update_layout(
            title="Training Metrics Comparison Across Runs",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_training_curves_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create training curves over epochs for selected runs"""
        fig = go.Figure()
        
        if df.empty:
            return fig.add_annotation(text="No data available", 
                                    xref="paper", yref="paper", 
                                    x=0.5, y=0.5, showarrow=False)
        
        colors = px.colors.qualitative.Set1
        
        for i, (run_index, run_data) in enumerate(df.iterrows()):
            # Get actual run_id from the data, not the index
            actual_run_id = run_data.get('run_id', str(run_index))
            run_name = run_data.get('tags.mlflow.runName', f'Run {str(actual_run_id)[:8]}')
            
            # Get metrics history for this run
            try:
                client = mlflow.tracking.MlflowClient()
                metrics_history = client.get_metric_history(run_data['run_id'], 'train_loss')
                
                if metrics_history:
                    epochs = [m.step for m in metrics_history]
                    losses = [m.value for m in metrics_history]
                    
                    fig.add_trace(go.Scatter(
                        x=epochs,
                        y=losses,
                        mode='lines+markers',
                        name=run_name,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6)
                    ))
            except Exception as e:
                logger.warning(f"Could not fetch metrics history for {run_name}: {e}")
        
        fig.update_layout(
            title="Training Loss Curves",
            xaxis_title="Epoch",
            yaxis_title="Training Loss",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_hyperparameter_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create hyperparameter comparison visualization"""
        if df.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper", 
                                            x=0.5, y=0.5, showarrow=False)
        
        # Extract key hyperparameters
        param_cols = [col for col in df.columns if col.startswith('params.')]
        if not param_cols:
            return go.Figure().add_annotation(text="No parameters found", 
                                            xref="paper", yref="paper", 
                                            x=0.5, y=0.5, showarrow=False)
        
        # Create parallel coordinates plot
        dimensions = []
        for col in param_cols[:8]:  # Limit to 8 parameters for readability
            if df[col].dtype in ['int64', 'float64']:
                dimensions.append(dict(
                    label=col.replace('params.', '').replace('config_', ''),
                    values=df[col].fillna(0)
                ))
        
        if dimensions:
            fig = go.Figure(data=go.Parcoords(
                line=dict(color=df.index, colorscale='Viridis'),
                dimensions=dimensions
            ))
            
            fig.update_layout(
                title="Hyperparameter Comparison",
                height=400
            )
        else:
            fig = go.Figure().add_annotation(text="No numeric parameters found", 
                                           xref="paper", yref="paper", 
                                           x=0.5, y=0.5, showarrow=False)
        
        return fig
    
    def create_performance_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary table of run performance"""
        if df.empty:
            return pd.DataFrame()
        
        summary_cols = ['tags.mlflow.runName', 'start_time', 'duration_minutes', 'status']
        metric_cols = [col for col in df.columns if col.startswith('metrics.')]
        param_cols = ['params.project_id', 'params.serial_number', 'params.device']
        
        display_cols = summary_cols + metric_cols + param_cols
        available_cols = [col for col in display_cols if col in df.columns]
        
        summary_df = df[available_cols].copy()
        
        # Clean column names
        summary_df.columns = [col.replace('tags.mlflow.runName', 'Run Name')
                             .replace('params.', '')
                             .replace('metrics.', '')
                             .replace('_', ' ').title() 
                             for col in summary_df.columns]
        
        # Format duration
        if 'Duration Minutes' in summary_df.columns:
            summary_df['Duration Minutes'] = summary_df['Duration Minutes'].round(2)
        
        # Format start time
        if 'Start Time' in summary_df.columns:
            summary_df['Start Time'] = summary_df['Start Time'].dt.strftime('%Y-%m-%d %H:%M')
        
        return summary_df.round(4)

def create_streamlit_dashboard():
    """Create Streamlit dashboard interface"""
    st.set_page_config(
        page_title="Latent Neurolese MLflow Dashboard",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Latent Neurolese Training Dashboard")
    st.markdown("Compare and analyze your LNSP model training runs")
    
    # Initialize dashboard
    dashboard = MLflowDashboard()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Test MLflow connection and get experiments
    st.sidebar.write("🔍 **Connection Status**")
    
    try:
        # Test connection
        import requests
        response = requests.get("http://localhost:5005/health", timeout=5)
        st.sidebar.success("✅ MLflow server connected")
    except Exception as e:
        st.sidebar.error(f"❌ MLflow server not reachable: {str(e)[:50]}...")
        st.error("🚨 **MLflow Server Not Running**")
        st.markdown("""
        **To fix this issue:**
        1. Start your MLflow server: `mlflow server --host 0.0.0.0 --port 5005`
        2. Or check if it's running on a different port
        3. Make sure you have some training runs completed
        """)
        return
    
    # Get experiments
    experiments = dashboard.get_experiments()
    if not experiments:
        st.error("📭 **No MLflow experiments found**")
        st.markdown("""
        **Possible reasons:**
        1. No training runs have been completed yet
        2. MLflow server is running but empty
        3. Check if experiments exist at: http://localhost:5005
        
        **To create experiments:**
        - Run a training job: `./venv/bin/python3 -m app.utils.run_project inputs/projects/Project_*.json`
        """)
        return
    
    # Experiment selection
    exp_names = [exp['name'] for exp in experiments]
    selected_experiments = st.sidebar.multiselect(
        "Select Experiments",
        exp_names,
        default=exp_names  # Default to ALL experiments
    )
    
    # Max results
    max_results = st.sidebar.slider("Max Runs to Display", 10, 100, 50)
    
    # Refresh button
    if st.sidebar.button("Refresh Data"):
        try:
            st.rerun()  # New Streamlit API
        except AttributeError:
            try:
                st.experimental_rerun()  # Fallback for older versions
            except AttributeError:
                st.cache_data.clear()  # Alternative refresh method
    
    if not selected_experiments:
        st.warning("Please select at least one experiment")
        return
    
    # Get selected experiment IDs
    selected_exp_ids = [exp['id'] for exp in experiments if exp['name'] in selected_experiments]
    
    # Fetch data
    with st.spinner("Loading MLflow data..."):
        df = dashboard.get_runs_data(selected_exp_ids, max_results)
    
    if df.empty:
        st.warning("No runs found for selected experiments")
        return
    
    # Display metrics
    st.header("📊 Run Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", len(df))
    with col2:
        completed_runs = len(df[df['status'] == 'FINISHED'])
        st.metric("Completed Runs", completed_runs)
    with col3:
        avg_duration = df['duration_minutes'].mean()
        st.metric("Avg Duration (min)", f"{avg_duration:.1f}" if pd.notna(avg_duration) else "N/A")
    with col4:
        if 'metrics.train_loss' in df.columns:
            best_loss = df['metrics.train_loss'].min()
            st.metric("Best Train Loss", f"{best_loss:.4f}" if pd.notna(best_loss) else "N/A")
    
    # Main visualizations
    st.header("📈 Training Metrics Comparison")
    metrics_fig = dashboard.create_metrics_comparison_chart(df)
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    st.header("📉 Training Curves")
    curves_fig = dashboard.create_training_curves_chart(df)
    st.plotly_chart(curves_fig, use_container_width=True)
    
    st.header("🔧 Hyperparameter Analysis")
    param_fig = dashboard.create_hyperparameter_comparison(df)
    st.plotly_chart(param_fig, use_container_width=True)
    
    # Summary table
    st.header("📋 Run Details")
    summary_df = dashboard.create_performance_summary_table(df)
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True)
    
    # Export options
    st.header("💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Summary CSV"):
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"mlflow_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Download Full Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Full CSV",
                data=csv,
                file_name=f"mlflow_full_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    create_streamlit_dashboard()
