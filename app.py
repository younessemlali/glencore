import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import optimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="CommodiPhys - Trading Physique",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #2E4057, #048A81);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #048A81;
}
.stMetric {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #e9ecef;
}
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header"><h1>⚒️ CommodiPhys - Trading Physique des Matières Premières</h1><p>Modèles physiques appliqués au négoce de métaux, énergie et agriculture</p></div>', unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("🏭 Navigation Trading")
section = st.sidebar.selectbox(
    "Module d'Analyse",
    [
        "🔥 Thermodynamique des Prix Énergie",
        "⚡ Modèle de Diffusion Métaux",
        "🌊 Dynamique des Flux Logistiques",
        "📊 Optimisation Portfolio Physique",
        "🎯 Arbitrage Géographique"
    ]
)

# Dictionnaire des commodités simulées (données fictives pour la démo)
COMMODITIES_DATA = {
    'Cuivre': {'price': 8500, 'volatility': 0.25, 'trend': 0.05},
    'Aluminium': {'price': 2100, 'volatility': 0.20, 'trend': 0.02}, 
    'Zinc': {'price': 2800, 'volatility': 0.30, 'trend': -0.01},
    'Pétrole Brent': {'price': 85, 'volatility': 0.35, 'trend': 0.08},
    'Pétrole WTI': {'price': 80, 'volatility': 0.33, 'trend': 0.07},
    'Gaz Naturel': {'price': 3.2, 'volatility': 0.45, 'trend': 0.12},
    'Or': {'price': 2050, 'volatility': 0.15, 'trend': 0.03},
    'Argent': {'price': 25, 'volatility': 0.28, 'trend': 0.04},
    'Blé': {'price': 550, 'volatility': 0.22, 'trend': 0.01},
    'Maïs': {'price': 420, 'volatility': 0.24, 'trend': 0.02}
}

def generate_price_series(base_price, volatility, trend, days=252):
    """Génère une série de prix simulée"""
    np.random.seed(42)
    returns = np.random.normal(trend/252, volatility/np.sqrt(252), days)
    prices = base_price * np.cumprod(1 + returns)
    return prices

def ornstein_uhlenbeck_simulation(S0, theta, mu, sigma, T, N, M):
    """Processus d'Ornstein-Uhlenbeck pour mean reversion des commodités"""
    dt = T/N
    paths = np.zeros((M, N+1))
    paths[:, 0] = S0
    
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt), M)
        paths[:, i] = paths[:, i-1] + theta * (mu - paths[:, i-1]) * dt + sigma * dW
    
    return paths

def jump_diffusion_model(S0, mu, sigma, lambda_jump, jump_mean, jump_std, T, N, M):
    """Modèle de diffusion avec sauts pour les chocs de supply/demand"""
    dt = T/N
    paths = np.zeros((M, N+1))
    paths[:, 0] = S0
    
    for i in range(1, N+1):
        # Composante diffusion
        dW = np.random.normal(0, np.sqrt(dt), M)
        diffusion = (mu - 0.5*sigma**2)*dt + sigma*dW
        
        # Composante sauts (événements géopolitiques, arrêts de production)
        jumps = np.random.poisson(lambda_jump*dt, M)
        jump_sizes = np.random.normal(jump_mean, jump_std, M) * jumps
        
        paths[:, i] = paths[:, i-1] * np.exp(diffusion + jump_sizes)
    
    return paths

def thermal_equilibrium_pricing(temperature_data, energy_prices, base_temp=20):
    """Modèle thermodynamique pour pricing de l'énergie"""
    temp_deviation = np.abs(temperature_data - base_temp)
    demand_multiplier = 1 + 0.02 * temp_deviation
    equilibrium_prices = energy_prices * demand_multiplier
    return equilibrium_prices, demand_multiplier

# Interface principale selon la section sélectionnée
if section == "🔥 Thermodynamique des Prix Énergie":
    st.header("🔥 Modèle Thermodynamique des Prix de l'Énergie")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres du Modèle")
        
        energy_commodity = st.selectbox(
            "Commodité Énergétique",
            ["Pétrole Brent", "Pétrole WTI", "Gaz Naturel"]
        )
        
        base_temp = st.slider("Température de référence (°C)", 15, 25, 20)
        sensitivity = st.slider("Sensibilité thermique (%/°C)", 1, 5, 2)
        days = st.slider("Période d'analyse (jours)", 30, 365, 90)
        
    with col2:
        # Génération des données simulées
        commodity_data = COMMODITIES_DATA[energy_commodity]
        base_price = commodity_data['price']
        
        # Simulation de données de température
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        temperatures = 15 + 10 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 3, days)
        
        # Prix simulés
        prices = generate_price_series(base_price, commodity_data['volatility'], commodity_data['trend'], days)
        
        # Calcul des prix d'équilibre thermodynamique
        equilibrium_prices, demand_mult = thermal_equilibrium_pricing(temperatures, prices, base_temp)
        
        # Graphique principal
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Prix vs Température", "Multiplicateur de Demande"],
            specs=[[{"secondary_y": True}], [{}]]
        )
        
        # Prix et température
        fig.add_trace(
            go.Scatter(x=dates, y=prices, name="Prix Réel", line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=equilibrium_prices, name="Prix Équilibre Thermique", 
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=temperatures, name="Température (°C)", 
                      line=dict(color='orange'), yaxis="y2"),
            row=1, col=1, secondary_y=True
        )
        
        # Multiplicateur de demande
        fig.add_trace(
            go.Scatter(x=dates, y=demand_mult, name="Multiplicateur Demande", 
                      fill='tonexty', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title=f"Analyse Thermodynamique - {energy_commodity}")
        fig.update_yaxes(title_text="Prix", row=1, col=1)
        fig.update_yaxes(title_text="Température °C", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Multiplicateur", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métriques clés
        st.subheader("📊 Métriques Thermodynamiques")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            avg_premium = np.mean(equilibrium_prices - prices)
            st.metric("Prime Thermique Moyenne", f"${avg_premium:.2f}")
        
        with col_b:
            max_demand = np.max(demand_mult)
            st.metric("Pic de Demande", f"{max_demand:.2f}x")
        
        with col_c:
            temp_volatility = np.std(temperatures)
            st.metric("Volatilité Température", f"{temp_volatility:.1f}°C")
        
        with col_d:
            correlation = np.corrcoef(temperatures, prices)[0,1]
            st.metric("Corrélation T°/Prix", f"{correlation:.3f}")

elif section == "⚡ Modèle de Diffusion Métaux":
    st.header("⚡ Modèle de Diffusion avec Sauts - Métaux")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres du Modèle")
        
        metal = st.selectbox("Métal", ["Cuivre", "Aluminium", "Zinc", "Or", "Argent"])
        
        # Paramètres du modèle
        commodity_data = COMMODITIES_DATA[metal]
        S0 = st.number_input("Prix Initial", value=float(commodity_data['price']), min_value=0.1)
        mu = st.slider("Drift (μ)", -0.2, 0.3, 0.05)
        sigma = st.slider("Volatilité (σ)", 0.1, 0.8, 0.25)
        lambda_jump = st.slider("Fréquence des Sauts", 0.0, 5.0, 1.0)
        jump_mean = st.slider("Taille Moyenne des Sauts", -0.2, 0.2, 0.0)
        jump_std = st.slider("Volatilité des Sauts", 0.01, 0.3, 0.1)
        
        T = st.slider("Horizon (années)", 0.25, 2.0, 1.0)
        n_sims = st.slider("Nombre de Simulations", 100, 1000, 500)
        
        if st.button("🚀 Lancer Simulation"):
            st.session_state.run_metal_sim = True
    
    with col2:
        if hasattr(st.session_state, 'run_metal_sim') and st.session_state.run_metal_sim:
            # Simulation du modèle de diffusion avec sauts
            N = int(T * 252)
            paths = jump_diffusion_model(S0, mu, sigma, lambda_jump, jump_mean, jump_std, T, N, n_sims)
            
            # Graphique des trajectoires
            time_grid = np.linspace(0, T, N+1)
            
            fig = go.Figure()
            
            # Quelques trajectoires échantillons
            for i in range(min(50, n_sims)):
                fig.add_trace(go.Scatter(
                    x=time_grid, y=paths[i], 
                    mode='lines', 
                    line=dict(width=0.5, color='lightblue'),
                    showlegend=False
                ))
            
            # Moyenne et percentiles
            mean_path = np.mean(paths, axis=0)
            p5 = np.percentile(paths, 5, axis=0)
            p95 = np.percentile(paths, 95, axis=0)
            
            fig.add_trace(go.Scatter(
                x=time_grid, y=mean_path,
                mode='lines', name='Moyenne',
                line=dict(width=3, color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=time_grid, y=p95,
                mode='lines', name='95e percentile',
                line=dict(width=2, color='green', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=time_grid, y=p5,
                mode='lines', name='5e percentile',
                line=dict(width=2, color='green', dash='dash'),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title=f"Simulation Prix {metal} - Modèle de Diffusion avec Sauts",
                xaxis_title="Temps (années)",
                yaxis_title="Prix",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            final_prices = paths[:, -1]
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Prix Final Moyen", f"${np.mean(final_prices):.2f}")
            with col_b:
                st.metric("Volatilité Réalisée", f"{np.std(final_prices)/S0*100:.1f}%")
            with col_c:
                prob_profit = np.mean(final_prices > S0) * 100
                st.metric("Probabilité de Gain", f"{prob_profit:.1f}%")
            
            # Distribution des prix finaux
            fig_hist = go.Figure(data=[go.Histogram(x=final_prices, nbinsx=50)])
            fig_hist.update_layout(
                title="Distribution des Prix Finaux",
                xaxis_title="Prix Final",
                yaxis_title="Fréquence"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

elif section == "🌊 Dynamique des Flux Logistiques":
    st.header("🌊 Optimisation des Flux Logistiques")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres Logistiques")
        
        storage_capacity = st.number_input("Capacité Stockage (tonnes)", 1000, 50000, 10000)
        storage_cost = st.number_input("Coût Stockage ($/tonne/mois)", 1, 50, 10)
        
        st.subheader("Prévisions de Demande")
        n_months = st.slider("Horizon (mois)", 3, 12, 6)
        base_demand = st.number_input("Demande de Base (tonnes/mois)", 1000, 10000, 5000)
        seasonality = st.slider("Saisonnalité (%)", 0, 50, 20)
        
        base_price = st.number_input("Prix de Base ($/tonne)", 1000, 10000, 5000)
        price_volatility = st.slider("Volatilité Prix (%)", 5, 30, 15)
    
    with col2:
        # Génération des données
        months = np.arange(1, n_months + 1)
        
        # Demande avec saisonnalité
        seasonal_pattern = 1 + (seasonality/100) * np.sin(2 * np.pi * months / 12)
        demand_forecast = base_demand * seasonal_pattern
        
        # Prix avec volatilité
        np.random.seed(42)
        price_changes = np.random.normal(0, price_volatility/100, n_months)
        prices = base_price * np.cumprod(1 + price_changes)
        
        purchase_costs = prices * 0.98
        selling_prices = prices * 1.02
        
        # Graphique des prévisions
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Demande Prévisionnelle", "Évolution des Prix"],
            specs=[[{}], [{}]]
        )
        
        fig.add_trace(
            go.Scatter(x=months, y=demand_forecast, 
                      mode='lines+markers', name='Demande',
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=months, y=prices, 
                      mode='lines+markers', name='Prix Marché',
                      line=dict(color='red', width=3)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=months, y=purchase_costs, 
                      mode='lines', name='Coût Achat',
                      line=dict(color='green', dash='dash')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=months, y=selling_prices, 
                      mode='lines', name='Prix Vente',
                      line=dict(color='orange', dash='dash')),
            row=2, col=1
        )
        
        fig.update_layout(height=500, title="Analyse des Flux Logistiques")
        fig.update_xaxes(title_text="Mois", row=2, col=1)
        fig.update_yaxes(title_text="Tonnes", row=1, col=1)
        fig.update_yaxes(title_text="Prix ($/tonne)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calcul des métriques
        total_demand = np.sum(demand_forecast)
        avg_price = np.mean(prices)
        price_trend = (prices[-1] - prices[0]) / prices[0] * 100
        
        st.subheader("📊 Métriques Logistiques")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Demande Totale", f"{total_demand:,.0f} t")
        
        with col_b:
            st.metric("Prix Moyen", f"${avg_price:,.0f}/t")
        
        with col_c:
            utilization = min(total_demand / (storage_capacity * n_months) * 100, 100)
            st.metric("Utilisation Stockage", f"{utilization:.1f}%")
        
        with col_d:
            st.metric("Tendance Prix", f"{price_trend:+.1f}%")
        
        # Recommandations
        st.subheader("🎯 Recommandations Stratégiques")
        
        if price_trend > 10:
            st.success("📈 **Stratégie**: Accumulation recommandée - Tendance haussière forte")
        elif price_trend < -10:
            st.warning("📉 **Stratégie**: Déstockage recommandé - Tendance baissière")
        else:
            st.info("⚖️ **Stratégie**: Maintien des stocks - Marché stable")

elif section == "📊 Optimisation Portfolio Physique":
    st.header("📊 Optimisation de Portfolio Physique Multi-Commodités")
    
    selected_commodities = st.multiselect(
        "Sélectionner les Commodités",
        list(COMMODITIES_DATA.keys()),
        default=["Cuivre", "Or", "Pétrole Brent"]
    )
    
    if len(selected_commodities) >= 2:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres d'Optimisation")
            
            total_capital = st.number_input("Capital Total ($M)", 10, 1000, 100)
            min_allocation = st.slider("Allocation Minimale (%)", 0, 20, 5)
            max_allocation = st.slider("Allocation Maximale (%)", 50, 100, 40)
            
            objective = st.radio(
                "Objectif",
                ["Maximiser Rendement", "Minimiser Risque", "Ratio Sharpe"]
            )
        
        with col2:
            # Génération de données de rendements simulées
            returns_data = {}
            
            for commodity in selected_commodities:
                data = COMMODITIES_DATA[commodity]
                prices = generate_price_series(data['price'], data['volatility'], data['trend'], 252)
                returns = np.diff(np.log(prices))
                returns_data[commodity] = returns
            
            # Création du DataFrame de rendements
            returns_df = pd.DataFrame(returns_data)
            
            # Calcul des statistiques
            mean_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252
            
            # Simulation Monte Carlo pour l'optimisation
            n_portfolios = 5000
            np.random.seed(42)
            n_assets = len(selected_commodities)
            
            results = np.zeros((4, n_portfolios))
            
            for i in range(n_portfolios):
                weights = np.random.random(n_assets)
                weights = weights / np.sum(weights)
                
                # Contraintes d'allocation
                if np.any(weights < min_allocation/100) or np.any(weights > max_allocation/100):
                    continue
                
                portfolio_return = np.sum(weights * mean_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                
                results[0, i] = portfolio_return
                results[1, i] = portfolio_risk
                results[2, i] = sharpe_ratio
                results[3, i] = i
            
            # Filtrage des résultats valides
            valid_indices = results[3, :] != 0
            valid_results = results[:, valid_indices]
            
            if valid_results.shape[1] > 0:
                # Sélection du portfolio optimal
                if objective == "Maximiser Rendement":
                    optimal_idx = np.argmax(valid_results[0, :])
                elif objective == "Minimiser Risque":
                    optimal_idx = np.argmin(valid_results[1, :])
                else:
                    optimal_idx = np.argmax(valid_results[2, :])
                
                optimal_return = valid_results[0, optimal_idx]
                optimal_risk = valid_results[1, optimal_idx]
                optimal_sharpe = valid_results[2, optimal_idx]
                
                # Graphique de la frontière efficiente
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=valid_results[1, :], 
                    y=valid_results[0, :],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=valid_results[2, :],
                        colorscale='Viridis',
                        colorbar=dict(title="Ratio Sharpe"),
                        opacity=0.6
                    ),
                    name='Portfolios'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[optimal_risk], 
                    y=[optimal_return],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Portfolio Optimal'
                ))
                
                fig.update_layout(
                    title="Frontière Efficiente - Commodités",
                    xaxis_title="Risque (Volatilité Annuelle)",
                    yaxis_title="Rendement Attendu (Annuel)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Métriques du portfolio optimal
                st.subheader("🎯 Portfolio Optimal")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Rendement Attendu", f"{optimal_return*100:.2f}%")
                with col_b:
                    st.metric("Risque (Volatilité)", f"{optimal_risk*100:.2f}%")
                with col_c:
                    st.metric("Ratio Sharpe", f"{optimal_sharpe:.3f}")
                
                # Matrice de corrélation
                st.subheader("🔗 Matrice de Corrélation")
                corr_matrix = returns_df.corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(color="Corrélation"),
                    color_continuous_scale="RdBu_r"
                )
                fig_corr.update_layout(title="Corrélations entre Commodités")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.error("Aucun portfolio valide trouvé avec les contraintes spécifiées")
    else:
        st.warning("Veuillez sélectionner au moins 2 commodités pour l'optimisation de portfolio")

elif section == "🎯 Arbitrage Géographique":
    st.header("🎯 Arbitrage Géographique et Différentiels de Prix")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres d'Arbitrage")
        
        commodity = st.selectbox("Commodité", ["Pétrole Brent", "Pétrole WTI", "Gaz Naturel"])
        
        transport_cost = st.number_input("Coût Transport ($/unité)", 0.5, 20.0, 5.0)
        transport_time = st.slider("Temps Transport (jours)", 1, 30, 7)
        storage_cost_daily = st.number_input("Coût Stockage ($/unité/jour)", 0.01, 1.0, 0.1)
        
        financing_rate = st.slider("Taux de Financement (%)", 0.0, 10.0, 3.0) / 100
        min_profit_margin = st.slider("Marge Minimum (%)", 0.5, 5.0, 1.0) / 100
        
    with col2:
        if commodity in ["Pétrole Brent", "Pétrole WTI"]:
            # Simulation des différentiels Brent vs WTI
            np.random.seed(42)
            days = 60
            dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
            
            brent_data = COMMODITIES_DATA["Pétrole Brent"]
            wti_data = COMMODITIES_DATA["Pétrole WTI"]
            
            brent_prices = generate_price_series(brent_data['price'], brent_data['volatility'], brent_data['trend'], days)
            wti_prices = generate_price_series(wti_data['price'], wti_data['volatility'], wti_data['trend'], days)
            
            # Calcul des opportunités d'arbitrage
            price_differential = brent_prices - wti_prices
            
            total_costs = transport_cost + storage_cost_daily * transport_time + financing_rate * wti_prices * (transport_time/365)
            
            gross_profit = price_differential
            net_profit = gross_profit - total_costs
            arbitrage_opportunities = net_profit > (wti_prices * min_profit_margin)
            
            # Graphique principal
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=["Prix Brent vs WTI", "Différentiel de Prix", "Opportunités d'Arbitrage"],
                vertical_spacing=0.08
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=brent_prices, name="Brent", line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=wti_prices, name="WTI", line=dict(color='red')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=price_differential, name="Différentiel Brent-WTI", 
                          fill='tonexty', line=dict(color='green')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=total_costs, name="Coûts d'Arbitrage", 
                          line=dict(color='orange', dash='dash')),
                row=2, col=1
            )
            
            # Profit net
            colors = ['green' if profit > 0 else 'red' for profit in net_profit]
            fig.add_trace(
                go.Bar(x=dates, y=net_profit, name="Profit Net", 
                      marker_color=colors, opacity=0.7),
                row=3, col=1
            )
            
            fig.update_layout(height=700, title="Analyse d'Arbitrage Géographique - Pétrole")
            fig.update_yaxes(title_text="Prix ($/baril)", row=1, col=1)
            fig.update_yaxes(title_text="Différentiel ($/baril)", row=2, col=1)
            fig.update_yaxes(title_text="Profit Net ($/baril)", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques d'arbitrage
            st.subheader("📊 Statistiques d'Arbitrage")
            
            profitable_days = np.sum(arbitrage_opportunities)
            total_potential_profit = np.sum(net_profit[net_profit > 0])
            avg_profit_per_opportunity = total_potential_profit / max(profitable_days, 1)
            success_rate = profitable_days / days * 100
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Jours Profitables", f"{profitable_days}/{days}")
            
            with col_b:
                st.metric("Taux de Succès", f"{success_rate:.1f}%")
            
            with col_c:
                st.metric("Profit Total Potentiel", f"${total_potential_profit:.2f}")
            
            with col_d:
                st.metric("Profit Moyen/Opportunité", f"${avg_profit_per_opportunity:.2f}")
            
            # Analyse des risques
            st.subheader("⚠️ Analyse des Risques")
            
            max_loss = np.min(net_profit)
            volatility_differential = np.std(price_differential)
            correlation_coeff = np.corrcoef(brent_prices, wti_prices)[0,1]
            
            col_risk1, col_risk2 = st.columns(2)
            
            with col_risk1:
                st.markdown(f"""
                **Risques Identifiés:**
                - Perte Maximum: ${max_loss:.2f}/baril
                - Volatilité Différentiel: ${volatility_differential:.2f}
                - Corrélation Prix: {correlation_coeff:.3f}
                """)
            
            with col_risk2:
                st.markdown(f"""
                **Facteurs de Risque:**
                - Changements géopolitiques
                - Disruptions logistiques
                - Variations de qualité
                - Risques de contrepartie
                """)
            
            # Recommandations
            if success_rate > 60:
                st.success("✅ **Recommandation**: Stratégie d'arbitrage attractive avec taux de succès élevé")
            elif success_rate > 40:
                st.warning("⚠️ **Recommandation**: Stratégie modérément attractive, surveiller les coûts")
            else:
                st.error("❌ **Recommandation**: Stratégie peu attractive, revoir les paramètres")

        elif commodity == "Gaz Naturel":
            st.subheader("🌍 Arbitrage Gaz Naturel - Différentiels Régionaux")
            
            # Simulation des prix régionaux
            regions = ["Henry Hub (US)", "NBP (UK)", "TTF (NL)", "JKM (Asie)"]
            base_prices = [3.5, 8.2, 7.8, 12.5]  # $/MMBtu
            
            np.random.seed(42)
            days = 45
            dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
            
            regional_prices = {}
            for i, region in enumerate(regions):
                volatility = 0.03
                trend = 0.0
                prices = generate_price_series(base_prices[i], volatility, trend, days)
                regional_prices[region] = prices
            
            # Création du DataFrame
            prices_df = pd.DataFrame(regional_prices, index=dates)
            
            # Graphique des prix régionaux
            fig = go.Figure()
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, region in enumerate(regions):
                fig.add_trace(go.Scatter(
                    x=dates, y=prices_df[region],
                    mode='lines', name=region,
                    line=dict(color=colors[i], width=2)
                ))
            
            fig.update_layout(
                title="Prix Gaz Naturel par Région",
                xaxis_title="Date",
                yaxis_title="Prix ($/MMBtu)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Matrice des spreads
            st.subheader("📈 Matrice des Spreads")
            
            current_prices = prices_df.iloc[-1]
            spread_matrix = np.zeros((len(regions), len(regions)))
            
            for i in range(len(regions)):
                for j in range(len(regions)):
                    spread_matrix[i][j] = current_prices.iloc[i] - current_prices.iloc[j]
            
            spread_df = pd.DataFrame(spread_matrix, index=regions, columns=regions)
            
            fig_spread = px.imshow(
                spread_df,
                labels=dict(color="Spread ($/MMBtu)"),
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig_spread.update_layout(title="Matrice des Spreads Actuels")
            st.plotly_chart(fig_spread, use_container_width=True)
            
            # Opportunités d'arbitrage identifiées
            st.subheader("🎯 Opportunités Identifiées")
            
            # Calcul des meilleures opportunités
            opportunities = []
            for i in range(len(regions)):
                for j in range(len(regions)):
                    if i != j:
                        spread = current_prices.iloc[j] - current_prices.iloc[i]
                        # Estimation simple des coûts de transport
                        if "Asie" in regions[j]:
                            transport_cost_est = 2.0
                        elif "US" in regions[i] and ("UK" in regions[j] or "NL" in regions[j]):
                            transport_cost_est = 1.5
                        else:
                            transport_cost_est = 1.0
                        
                        net_spread = spread - transport_cost_est
                        if net_spread > 0.5:  # Seuil de rentabilité
                            opportunities.append({
                                'Achat': regions[i],
                                'Vente': regions[j],
                                'Spread Brut': spread,
                                'Coût Transport': transport_cost_est,
                                'Spread Net': net_spread,
                                'ROI Estimé': (net_spread / current_prices.iloc[i]) * 100
                            })
            
            if opportunities:
                opportunities_df = pd.DataFrame(opportunities)
                opportunities_df = opportunities_df.sort_values('Spread Net', ascending=False)
                
                st.dataframe(
                    opportunities_df.style.format({
                        'Spread Brut': '${:.2f}',
                        'Coût Transport': '${:.2f}',
                        'Spread Net': '${:.2f}',
                        'ROI Estimé': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("Aucune opportunité d'arbitrage rentable identifiée actuellement")

# Section informations et déploiement
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Guide de Déploiement")

with st.sidebar.expander("🚀 Déploiement GitHub → Streamlit"):
    st.markdown("""
    **Étapes de déploiement:**
    
    1. **Créer le repository GitHub**
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    git remote add origin <your-repo-url>
    git push -u origin main
    ```
    
    2. **Créer requirements.txt**
    ```
    streamlit>=1.28.0
    numpy>=1.24.0
    pandas>=2.0.0
    plotly>=5.15.0
    scipy>=1.11.0
    ```
    
    3. **Connecter à Streamlit Cloud**
    - Aller sur share.streamlit.io
    - Connecter votre GitHub
    - Sélectionner le repository
    - Déployer automatiquement
    """)

with st.sidebar.expander("⚙️ Structure du Projet"):
    st.markdown("""
    ```
    commodiphys/
    ├── app.py                 # Application principale
    ├── requirements.txt       # Dépendances
    ├── .streamlit/
    │   └── config.toml       # Configuration
    └── README.md             # Documentation
    ```
    """)

with st.sidebar.expander("📊 Fonctionnalités"):
    st.markdown("""
    **Modules Disponibles:**
    - 🔥 Thermodynamique des Prix Énergie
    - ⚡ Modèle de Diffusion Métaux
    - 🌊 Dynamique des Flux Logistiques
    - 📊 Optimisation Portfolio Physique
    - 🎯 Arbitrage Géographique
    
    **Modèles Physiques:**
    - Processus d'Ornstein-Uhlenbeck
    - Diffusion avec sauts
    - Équilibre thermodynamique
    - Optimisation de Markowitz
    """)

# Footer avec informations
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
<p><strong>CommodiPhys v1.0</strong> - Application de Trading Physique des Matières Premières</p>
<p>Modèles physiques appliqués • Optimisation quantitative • Analyse des risques</p>
<p>🔗 <em>Déployez sur GitHub → Streamlit Cloud pour une version live</em></p>
</div>
""", unsafe_allow_html=True)
