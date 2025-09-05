import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import optimize, integrate
from scipy.stats import norm, t
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="CommodiPhys - Trading Physique",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal avec style industriel
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
</style>
""", unsafe_allow_html=True)

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

# Dictionnaire des commodités avec leurs tickers
COMMODITIES = {
    'Cuivre': 'HG=F',
    'Aluminium': 'ALI=F', 
    'Zinc': 'ZN=F',
    'Pétrole Brent': 'BZ=F',
    'Pétrole WTI': 'CL=F',
    'Gaz Naturel': 'NG=F',
    'Or': 'GC=F',
    'Argent': 'SI=F',
    'Blé': 'ZW=F',
    'Maïs': 'ZC=F',
    'Charbon': 'MTF=F'
}

@st.cache_data
def get_commodity_data(ticker, period="2y"):
    """Récupération des données de commodités"""
    try:
        commodity = yf.Ticker(ticker)
        data = commodity.history(period=period)
        return data
    except:
        return None

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

def convenience_yield_model(spot_price, futures_prices, time_to_maturity, risk_free_rate):
    """Calcul du convenience yield pour stockage physique"""
    convenience_yields = []
    for i, (future_price, T) in enumerate(zip(futures_prices, time_to_maturity)):
        if T > 0:
            cy = risk_free_rate + (1/T) * np.log(spot_price/future_price)
            convenience_yields.append(cy)
        else:
            convenience_yields.append(0)
    return np.array(convenience_yields)

def storage_cost_optimization(demand_forecast, storage_capacity, storage_cost_per_unit, 
                            purchase_prices, selling_prices):
    """Optimisation des coûts de stockage physique"""
    n_periods = len(demand_forecast)
    
    # Variables de décision: achats, ventes, stock
    from scipy.optimize import minimize
    
    def objective(x):
        purchases = x[:n_periods]
        sales = x[n_periods:2*n_periods]
        storage = x[2*n_periods:3*n_periods]
        
        # Coût total = coûts d'achat + coûts de stockage - revenus de vente
        total_cost = (np.sum(purchases * purchase_prices) + 
                     np.sum(storage * storage_cost_per_unit) - 
                     np.sum(sales * selling_prices))
        return total_cost
    
    # Contraintes
    constraints = []
    bounds = []
    
    # Bornes pour toutes les variables (non-négatives)
    for i in range(3*n_periods):
        bounds.append((0, None))
    
    # Contrainte de capacité de stockage
    for i in range(n_periods):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: storage_capacity - x[2*n_periods + i]
        })
    
    # Point de départ
    x0 = np.ones(3*n_periods) * 10
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    except:
        return None

def thermal_equilibrium_pricing(temperature_data, energy_prices, base_demand):
    """Modèle thermodynamique pour pricing de l'énergie"""
    # Relation entre température et demande énergétique (chauffage/climatisation)
    temp_deviation = np.abs(temperature_data - 20)  # Température de confort 20°C
    demand_multiplier = 1 + 0.02 * temp_deviation  # 2% d'augmentation par degré d'écart
    
    # Prix d'équilibre basé sur la demande thermodynamique
    equilibrium_prices = energy_prices * demand_multiplier
    
    return equilibrium_prices, demand_multiplier

# Interface principale selon la section sélectionnée
if section == "🔥 Thermodynamique des Prix Énergie":
    st.header("🔥 Modèle Thermodynamique des Prix de l'Énergie")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres du Modèle")
        
        # Sélection de la commodité énergétique
        energy_commodity = st.selectbox(
            "Commodité Énergétique",
            ["Pétrole Brent", "Pétrole WTI", "Gaz Naturel", "Charbon"]
        )
        
        # Paramètres thermodynamiques
        base_temp = st.slider("Température de référence (°C)", 15, 25, 20)
        sensitivity = st.slider("Sensibilité thermique (%/°C)", 1, 5, 2)
        
        # Simulation de données météo
        days = st.slider("Période d'analyse (jours)", 30, 365, 90)
        
    with col2:
        # Récupération des données
        ticker = COMMODITIES[energy_commodity]
        data = get_commodity_data(ticker, "1y")
        
        if data is not None and len(data) > 0:
            # Simulation de données de température
            np.random.seed(42)
            dates = pd.date_range(start=data.index[-days], periods=days, freq='D')
            temperatures = 15 + 10 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 3, days)
            
            # Calcul des prix d'équilibre thermodynamique
            recent_prices = data['Close'].iloc[-days:].values
            if len(recent_prices) < days:
                recent_prices = np.repeat(recent_prices[-1], days)
            
            equilibrium_prices, demand_mult = thermal_equilibrium_pricing(
                temperatures, recent_prices, base_demand=100
            )
            
            # Graphique principal
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Prix vs Température", "Multiplicateur de Demande"],
                specs=[[{"secondary_y": True}], [{}]]
            )
            
            # Prix et température
            fig.add_trace(
                go.Scatter(x=dates, y=recent_prices, name="Prix Réel", line=dict(color='blue')),
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
            fig.update_yaxes(title_text="Prix USD", row=1, col=1)
            fig.update_yaxes(title_text="Température °C", secondary_y=True, row=1, col=1)
            fig.update_yaxes(title_text="Multiplicateur", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Métriques clés
            st.subheader("📊 Métriques Thermodynamiques")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                avg_premium = np.mean(equilibrium_prices - recent_prices)
                st.metric("Prime Thermique Moyenne", f"${avg_premium:.2f}")
            
            with col_b:
                max_demand = np.max(demand_mult)
                st.metric("Pic de Demande", f"{max_demand:.2f}x")
            
            with col_c:
                temp_volatility = np.std(temperatures)
                st.metric("Volatilité Température", f"{temp_volatility:.1f}°C")
            
            with col_d:
                correlation = np.corrcoef(temperatures, recent_prices)[0,1]
                st.metric("Corrélation T°/Prix", f"{correlation:.3f}")

elif section == "⚡ Modèle de Diffusion Métaux":
    st.header("⚡ Modèle de Diffusion avec Sauts - Métaux")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Paramètres du Modèle")
        
        metal = st.selectbox("Métal", ["Cuivre", "Aluminium", "Zinc", "Or", "Argent"])
        
        # Paramètres du modèle
        S0 = st.number_input("Prix Initial", value=100.0, min_value=0.1)
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
            N = int(T * 252)  # Pas journaliers
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
        
        # Paramètres de stockage
        storage_capacity = st.number_input("Capacité Stockage (tonnes)", 1000, 50000, 10000)
        storage_cost = st.number_input("Coût Stockage ($/tonne/mois)", 1, 50, 10)
        
        # Prévisions de demande
        st.subheader("Prévisions de Demande")
        n_months = st.slider("Horizon (mois)", 3, 12, 6)
        
        base_demand = st.number_input("Demande de Base (tonnes/mois)", 1000, 10000, 5000)
        seasonality = st.slider("Saisonnalité (%)", 0, 50, 20)
        
        # Prix
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
        
        # Optimisation simple (version déterministe)
        purchase_costs = prices * 0.98  # Légère décote à l'achat
        selling_prices = prices * 1.02  # Prime à la vente
        
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
        
        # Calcul des métriques de performance
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
        
        # Recommandations stratégiques
        st.subheader("🎯 Recommandations Stratégiques")
        
        if price_trend > 10:
            st.success("📈 **Stratégie**: Accumulation recommandée - Tendance haussière forte")
        elif price_trend < -10:
            st.warning("📉 **Stratégie**: Déstockage recommandé - Tendance baissière")
        else:
            st.info("⚖️ **Stratégie**: Maintien des stocks - Marché stable")
        
        # Analyse de la saisonnalité
        peak_month = np.argmax(demand_forecast) + 1
        low_month = np.argmin(demand_forecast) + 1
        
        st.write(f"**Saisonnalité**: Pic de demande en mois {peak_month}, creux en mois {low_month}")

elif section == "📊 Optimisation Portfolio Physique":
    st.header("📊 Optimisation de Portfolio Physique Multi-Commodités")
    
    # Sélection des commodités pour le portfolio
    selected_commodities = st.multiselect(
        "Sélectionner les Commodités",
        list(COMMODITIES.keys()),
        default=["Cuivre", "Or", "Pétrole Brent"]
    )
    
    if len(selected_commodities) >= 2:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres d'Optimisation")
            
            # Contraintes de capital
            total_capital = st.number_input("Capital Total ($M)", 10, 1000, 100)
            min_allocation = st.slider("Allocation Minimale (%)", 0, 20, 5)
            max_allocation = st.slider("Allocation Maximale (%)", 50, 100, 40)
            
            # Horizon d'investissement
            investment_horizon = st.selectbox("Horizon", ["1 mois", "3 mois", "6 mois", "1 an"])
            horizon_map = {"1 mois": "1mo", "3 mois": "3mo", "6 mois": "6mo", "1 an": "1y"}
            
            # Objectif d'optimisation
            objective = st.radio(
                "Objectif",
                ["Maximiser Rendement", "Minimiser Risque", "Ratio Sharpe"]
            )
        
        with col2:
            # Récupération des données pour toutes les commodités sélectionnées
            returns_data = {}
            prices_data = {}
            
            for commodity in selected_commodities:
                ticker = COMMODITIES[commodity]
                data = get_commodity_data(ticker, "2y")
                if data is not None and len(data) > 0:
                    prices_data[commodity] = data['Close']
                    returns_data[commodity] = data['Close'].pct_change().dropna()
            
            if len(returns_data) >= 2:
                # Création de la matrice de rendements
                returns_df = pd.DataFrame(returns_data)
                returns_df = returns_df.dropna()
                
                # Calcul des statistiques
                mean_returns = returns_df.mean() * 252  # Annualisé
                cov_matrix = returns_df.cov() * 252     # Annualisé
                
                # Optimisation de Markowitz simplifiée
                n_assets = len(selected_commodities)
                
                # Simulation Monte Carlo pour l'optimisation
                n_portfolios = 10000
                np.random.seed(42)
                
                results = np.zeros((4, n_portfolios))
                
                for i in range(n_portfolios):
                    # Génération de poids aléatoires
                    weights = np.random.random(n_assets)
                    weights = weights / np.sum(weights)  # Normalisation
                    
                    # Contraintes d'allocation
                    if np.any(weights < min_allocation/100) or np.any(weights > max_allocation/100):
                        continue
                    
                    # Calcul des métriques du portfolio
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
                    # Sélection du portfolio optimal selon l'objectif
                    if objective == "Maximiser Rendement":
                        optimal_idx = np.argmax(valid_results[0, :])
                    elif objective == "Minimiser Risque":
                        optimal_idx = np.argmin(valid_results[1, :])
                    else:  # Ratio Sharpe
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
                st.error("Données
