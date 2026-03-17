import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import base64
import urllib.request
import urllib.error
import unicodedata
import re
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import io
import copy

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🎾 ATP Tennis Value Betting",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
BETS_DIR = BASE_DIR / "bets"
BETS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STYLES CSS
# ============================================================================

st.markdown("""
<style>
    :root {
        --primary-green: #2E7D32;
        --primary-gold: #F9A825;
        --clay: #C75B12;
        --grass: #388E3C;
        --hard: #1565C0;
        --success-color: #4CAF50;
        --warning-color: #FFC107;
        --error-color: #F44336;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #2E7D32 0%, #F9A825 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 30px;
    }
    
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-box {
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4CAF50;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .bet-recommendation {
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid var(--success-color);
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.05) 100%);
    }
    
    .no-bet {
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #888;
        background: linear-gradient(135deg, rgba(150, 150, 150, 0.1) 0%, rgba(150, 150, 150, 0.05) 100%);
    }
    
    .surface-hard { color: #1565C0; font-weight: bold; }
    .surface-clay { color: #C75B12; font-weight: bold; }
    .surface-grass { color: #388E3C; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# ============================================================================

@st.cache_resource
def load_model():
    """Charge le modèle XGBoost et le scaler"""
    model = joblib.load(MODELS_DIR / "xgb_v2b_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler_v2b.pkl")
    return model, scaler

@st.cache_resource
def load_elo_ratings():
    """Charge les ratings Elo (global + surface)"""
    return joblib.load(MODELS_DIR / "elo_ratings.pkl")

@st.cache_resource
def load_model_config():
    """Charge la configuration du modèle"""
    return joblib.load(MODELS_DIR / "model_config.pkl")

@st.cache_resource
def load_player_stats():
    """Charge les stats des joueurs"""
    return joblib.load(MODELS_DIR / "player_stats.pkl")

@st.cache_data
def load_recent_matches():
    """Charge les matchs récents pour calcul de features"""
    return pd.read_csv(MODELS_DIR / "recent_matches.csv", parse_dates=['Date'])

@st.cache_data
def load_historical_data():
    """Charge le dataset complet pour les stats"""
    df = pd.read_csv(DATA_DIR / "atp_tennis.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# ============================================================================
# THE ODDS API — Récupération des événements et cotes en direct
# ============================================================================

# 🔐 Clé API encodée (même clé que l'app UFC)
_ENCODED_API_KEY = "MTI4NTcwMTFmZjI3MDcwYWYxZTI4NTc2MTZkYWM1YjQ="

# Mapping tournois ATP connus → métadonnées (enrichi)
# Les tournois NON listés ici seront quand même récupérés dynamiquement
TENNIS_SPORT_KEYS = {
    # Grand Slams
    'tennis_atp_aus_open_singles': {'name': 'Australian Open', 'series': 'Grand Slam', 'surface': 'Hard', 'best_of': 5},
    'tennis_atp_french_open': {'name': 'French Open', 'series': 'Grand Slam', 'surface': 'Clay', 'best_of': 5},
    'tennis_atp_wimbledon': {'name': 'Wimbledon', 'series': 'Grand Slam', 'surface': 'Grass', 'best_of': 5},
    'tennis_atp_us_open': {'name': 'US Open', 'series': 'Grand Slam', 'surface': 'Hard', 'best_of': 5},
    # Masters 1000
    'tennis_atp_indian_wells': {'name': 'Indian Wells', 'series': 'Masters 1000', 'surface': 'Hard', 'best_of': 3},
    'tennis_atp_miami_open': {'name': 'Miami Open', 'series': 'Masters 1000', 'surface': 'Hard', 'best_of': 3},
    'tennis_atp_monte_carlo_masters': {'name': 'Monte-Carlo', 'series': 'Masters 1000', 'surface': 'Clay', 'best_of': 3},
    'tennis_atp_madrid_open': {'name': 'Madrid Open', 'series': 'Masters 1000', 'surface': 'Clay', 'best_of': 3},
    'tennis_atp_italian_open': {'name': 'Italian Open', 'series': 'Masters 1000', 'surface': 'Clay', 'best_of': 3},
    'tennis_atp_canadian_open': {'name': 'Canadian Open', 'series': 'Masters 1000', 'surface': 'Hard', 'best_of': 3},
    'tennis_atp_cincinnati_open': {'name': 'Cincinnati Open', 'series': 'Masters 1000', 'surface': 'Hard', 'best_of': 3},
    'tennis_atp_shanghai_masters': {'name': 'Shanghai Masters', 'series': 'Masters 1000', 'surface': 'Hard', 'best_of': 3},
    'tennis_atp_paris_masters': {'name': 'Paris Masters', 'series': 'Masters 1000', 'surface': 'Hard', 'best_of': 3},
    # ATP 500
    'tennis_atp_dubai': {'name': 'Dubai', 'series': 'ATP500', 'surface': 'Hard', 'best_of': 3},
    'tennis_atp_qatar_open': {'name': 'Qatar Open', 'series': 'ATP500', 'surface': 'Hard', 'best_of': 3},
    'tennis_atp_china_open': {'name': 'China Open', 'series': 'ATP500', 'surface': 'Hard', 'best_of': 3},
}


def infer_tournament_info(sport_key, sport_title):
    """Infère les métadonnées d'un tournoi ATP non listé dans TENNIS_SPORT_KEYS.
    
    Utilise le nom et la clé pour deviner la surface, le type de tournoi, etc.
    """
    # Déjà connu ?
    if sport_key in TENNIS_SPORT_KEYS:
        return TENNIS_SPORT_KEYS[sport_key]
    
    title_lower = sport_title.lower() if sport_title else sport_key.lower()
    key_lower = sport_key.lower()
    
    # Détecter la série
    if 'grand slam' in title_lower or any(gs in key_lower for gs in ['aus_open', 'french_open', 'wimbledon', 'us_open']):
        series = 'Grand Slam'
        best_of = 5
    elif 'masters' in title_lower or 'masters' in key_lower:
        series = 'Masters 1000'
        best_of = 3
    else:
        series = 'ATP'
        best_of = 3
    
    # Détecter la surface par mots-clés dans le titre/clé
    clay_keywords = ['french', 'roland', 'rome', 'italian', 'madrid', 'monte_carlo',
                     'monte carlo', 'barcelona', 'hamburg', 'buenos_aires', 'rio',
                     'chile', 'santiago', 'cordoba', 'bastad', 'gstaad', 'kitzbuhel',
                     'umag', 'bucharest', 'lyon', 'geneva', 'parma', 'marrakech']
    grass_keywords = ['wimbledon', 'halle', 'queens', 'queen', 'eastbourne', 'stuttgart_grass',
                      's_hertogenbosch', 'mallorca', 'newport']
    
    surface = 'Hard'  # Default
    for kw in clay_keywords:
        if kw in key_lower or kw in title_lower:
            surface = 'Clay'
            break
    if surface == 'Hard':
        for kw in grass_keywords:
            if kw in key_lower or kw in title_lower:
                surface = 'Grass'
                break
    
    # Nom lisible
    name = sport_title.replace('ATP ', '').replace('WTA ', '') if sport_title else sport_key.replace('tennis_atp_', '').replace('_', ' ').title()
    
    return {'name': name, 'series': series, 'surface': surface, 'best_of': best_of}

def _decode_api_key():
    """Décode la clé API"""
    try:
        return base64.b64decode(_ENCODED_API_KEY).decode('utf-8')
    except Exception:
        return None

def get_odds_api_key():
    """Récupère la clé API (encodée, session, secrets ou env)"""
    key = _decode_api_key()
    if not key and 'temp_odds_api_key' in st.session_state and st.session_state.temp_odds_api_key:
        key = st.session_state.temp_odds_api_key
    if not key:
        try:
            key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            pass
    if not key:
        key = os.environ.get("ODDS_API_KEY", "")
    if key:
        key = key.strip()
    return key

def fetch_active_tennis_sports(api_key=None):
    """Récupère TOUS les sports tennis ATP actifs sur The Odds API (dynamique).
    
    Ne se limite plus à TENNIS_SPORT_KEYS : tout sport dont la clé commence
    par 'tennis_atp_' et qui est actif est retourné.
    """
    if not api_key:
        api_key = get_odds_api_key()
    if not api_key:
        return [], "❌ Clé API manquante"
    
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Tennis-Predictor"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            sports = json.loads(resp.read().decode())
            remaining = resp.headers.get('x-requests-remaining', '?')
            used = resp.headers.get('x-requests-used', '?')
            
            # Récupérer TOUS les sports ATP actifs (pas seulement ceux de la liste)
            tennis = [s for s in sports if s.get('key', '').startswith('tennis_atp_') and s.get('active')]
            return tennis, f"✅ {len(tennis)} tournois actifs (Quota: {used}/{int(used)+int(remaining) if remaining != '?' and used != '?' else '?'})"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return [], "❌ Clé API invalide"
        elif e.code == 429:
            return [], "❌ Quota API dépassé"
        return [], f"❌ Erreur API: {e.code}"
    except Exception as e:
        return [], f"❌ Erreur: {str(e)}"

def fetch_tennis_odds(sport_key, api_key=None):
    """Récupère les cotes pour un tournoi tennis spécifique"""
    if not api_key:
        api_key = get_odds_api_key()
    if not api_key:
        return None, "❌ Clé API manquante"
    
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_url = f"{url}?{query_string}"
    
    try:
        req = urllib.request.Request(full_url, headers={"User-Agent": "Tennis-Predictor"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            remaining = resp.headers.get('x-requests-remaining', '?')
            used = resp.headers.get('x-requests-used', '?')
            
            events = []
            for event in data:
                home = event.get('home_team', '')
                away = event.get('away_team', '')
                commence = event.get('commence_time', '')
                
                # Chercher les bookmakers (priorité: pinnacle > betfair > unibet > winamax)
                bookmakers = event.get('bookmakers', [])
                priority = ['pinnacle', 'betfair_ex_eu', 'unibet', 'winamax', '1xbet']
                selected_book = None
                for prio in priority:
                    for book in bookmakers:
                        if book.get('key', '').lower().startswith(prio):
                            selected_book = book
                            break
                    if selected_book:
                        break
                if not selected_book and bookmakers:
                    selected_book = bookmakers[0]
                
                odds_home, odds_away = None, None
                bookmaker_name = 'N/A'
                if selected_book:
                    bookmaker_name = selected_book.get('title', 'Unknown')
                    for market in selected_book.get('markets', []):
                        if market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name') == home:
                                    odds_home = outcome.get('price')
                                elif outcome.get('name') == away:
                                    odds_away = outcome.get('price')
                
                # Toutes les cotes dispo (pour comparaison)
                all_bookmakers = []
                for book in bookmakers:
                    for market in book.get('markets', []):
                        if market.get('key') == 'h2h':
                            book_odds = {o.get('name'): o.get('price') for o in market.get('outcomes', [])}
                            all_bookmakers.append({
                                'name': book.get('title', ''),
                                'odds': book_odds
                            })
                
                events.append({
                    'home': home,
                    'away': away,
                    'commence_time': commence,
                    'odds_home': odds_home,
                    'odds_away': odds_away,
                    'bookmaker': bookmaker_name,
                    'all_bookmakers': all_bookmakers,
                    'event_id': event.get('id', ''),
                })
            
            return events, f"✅ {len(events)} matchs récupérés (Quota: {used}/{int(used)+int(remaining) if remaining != '?' and used != '?' else '?'})"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "❌ Clé API invalide"
        elif e.code == 429:
            return None, "❌ Quota API dépassé (500 req/mois gratuit)"
        return None, f"❌ Erreur API: {e.code}"
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"

# ============================================================================
# NAME MATCHING — Correspondance noms API ↔ noms modèle
# ============================================================================

def normalize_name(name):
    """Normalise un nom pour le matching"""
    if not name:
        return ""
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def match_api_name_to_model(api_name, model_players):
    """
    Trouve le nom du joueur dans le modèle correspondant au nom de l'API.
    
    L'API utilise 'Prénom Nom' (ex: 'Andrey Rublev')
    Le modèle utilise 'Nom I.' (ex: 'Rublev A.')
    """
    if not api_name:
        return None
    
    norm_api = normalize_name(api_name)
    api_parts = norm_api.split()
    
    if len(api_parts) < 2:
        return None
    
    api_last = api_parts[-1]  # Nom de famille
    api_first_initial = api_parts[0][0] if api_parts[0] else ''
    
    best_match = None
    best_score = 0
    
    for player in model_players:
        norm_player = normalize_name(player)
        player_parts = norm_player.split()
        
        if not player_parts:
            continue
        
        # Format modèle: "Nom I." → dernier mot est l'initiale, premiers mots le nom
        # Mais parfois c'est "De Minaur A." avec nom composé
        model_last = player_parts[0]  # Premier mot = nom de famille
        model_initial = player_parts[-1].replace('.', '') if len(player_parts) > 1 else ''
        
        # Score de matching
        score = 0
        
        # Match exact du nom de famille
        if api_last == model_last:
            score = 3
        elif api_last in model_last or model_last in api_last:
            score = 2
        else:
            # Essayer avec les autres parties (noms composés)
            model_name_parts = player_parts[:-1] if len(player_parts) > 1 else player_parts
            full_model_last = ' '.join(model_name_parts)
            api_full_last = ' '.join(api_parts[1:])
            if normalize_name(api_full_last) == full_model_last:
                score = 3
            elif api_last in full_model_last:
                score = 1.5
            else:
                continue
        
        # Bonus pour initiale
        if model_initial and api_first_initial == model_initial[0]:
            score += 1
        
        if score > best_score:
            best_score = score
            best_match = player
    
    return best_match if best_score >= 2 else None

def get_latest_rank(player, recent_matches):
    """Récupère le dernier classement et points connus d'un joueur"""
    latest = recent_matches.sort_values('Date', ascending=False)
    
    # Check as Player_1
    p1 = latest[latest['Player_1'] == player].head(1)
    if not p1.empty:
        r1 = p1['Rank_1'].values[0]
        pts1 = p1['Pts_1'].values[0]
        if pd.notna(r1) and r1 > 0:
            return int(r1), int(pts1) if pd.notna(pts1) else 1000
    
    # Check as Player_2
    p2 = latest[latest['Player_2'] == player].head(1)
    if not p2.empty:
        r2 = p2['Rank_2'].values[0]
        pts2 = p2['Pts_2'].values[0]
        if pd.notna(r2) and r2 > 0:
            return int(r2), int(pts2) if pd.notna(pts2) else 1000
    
    return 100, 1000  # Default fallback

# ============================================================================
# FONCTIONS DE CALCUL DES FEATURES
# ============================================================================

def get_player_elo(player, elo_data, surface=None):
    """Récupère l'Elo d'un joueur (global ou surface)"""
    global_elo = elo_data['global'].get(player, 1500)
    if surface and surface in elo_data['surface']:
        surface_elo = elo_data['surface'][surface].get(player, 1500)
    else:
        surface_elo = 1500
    return global_elo, surface_elo

def compute_h2h(player1, player2, recent_matches):
    """Calcule le H2H entre deux joueurs"""
    h2h = recent_matches[
        ((recent_matches['Player_1'] == player1) & (recent_matches['Player_2'] == player2)) |
        ((recent_matches['Player_1'] == player2) & (recent_matches['Player_2'] == player1))
    ]
    if len(h2h) == 0:
        return 0.5, 0
    
    p1_wins = ((h2h['Winner'] == player1)).sum()
    total = len(h2h)
    return p1_wins / total, total

def compute_recent_form(player, recent_matches, n_matches=5):
    """Calcule la forme récente (% victoires sur N derniers matchs)"""
    player_matches = recent_matches[
        (recent_matches['Player_1'] == player) | (recent_matches['Player_2'] == player)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    if len(player_matches) == 0:
        return 0.5
    
    wins = (player_matches['Winner'] == player).sum()
    return wins / len(player_matches)

def compute_surface_winrate(player, surface, recent_matches, months=12):
    """Calcule le win rate sur une surface donnée"""
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months)
    surf_matches = recent_matches[
        (recent_matches['Surface'] == surface) &
        (recent_matches['Date'] >= cutoff) &
        ((recent_matches['Player_1'] == player) | (recent_matches['Player_2'] == player))
    ]
    if len(surf_matches) == 0:
        return 0.5
    wins = (surf_matches['Winner'] == player).sum()
    return wins / len(surf_matches)

def compute_fatigue(player, recent_matches, days=14):
    """Calcule la fatigue (nombre de matchs récents)"""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = recent_matches[
        (recent_matches['Date'] >= cutoff) &
        ((recent_matches['Player_1'] == player) | (recent_matches['Player_2'] == player))
    ]
    return len(recent)

def build_feature_vector(player1, player2, surface, series, round_name, best_of,
                          rank1, rank2, pts1, pts2, elo_data, recent_matches, config):
    """Construit le vecteur de features pour la prédiction"""
    
    # Elo global + surface
    elo_p1_global, elo_p1_surf = get_player_elo(player1, elo_data, surface)
    elo_p2_global, elo_p2_surf = get_player_elo(player2, elo_data, surface)
    
    # H2H
    h2h_wr, h2h_total = compute_h2h(player1, player2, recent_matches)
    
    # Form
    form_5_p1 = compute_recent_form(player1, recent_matches, 5)
    form_5_p2 = compute_recent_form(player2, recent_matches, 5)
    form_10_p1 = compute_recent_form(player1, recent_matches, 10)
    form_10_p2 = compute_recent_form(player2, recent_matches, 10)
    
    # Surface win rate
    surf_wr_3m_p1 = compute_surface_winrate(player1, surface, recent_matches, 3)
    surf_wr_3m_p2 = compute_surface_winrate(player2, surface, recent_matches, 3)
    surf_wr_12m_p1 = compute_surface_winrate(player1, surface, recent_matches, 12)
    surf_wr_12m_p2 = compute_surface_winrate(player2, surface, recent_matches, 12)
    
    # Fatigue
    fatigue_p1 = compute_fatigue(player1, recent_matches)
    fatigue_p2 = compute_fatigue(player2, recent_matches)
    
    # Ranking features
    rank_diff = rank1 - rank2 if rank1 > 0 and rank2 > 0 else 0
    rank_ratio = rank1 / rank2 if rank1 > 0 and rank2 > 0 else 1
    pts_diff = pts1 - pts2
    
    # Surface encoding
    is_hard = 1 if surface == 'Hard' else 0
    is_clay = 1 if surface == 'Clay' else 0
    is_grass = 1 if surface == 'Grass' else 0
    
    # Best of
    best_of_5 = 1 if best_of == 5 else 0
    
    # Round encoding
    round_map = {
        '1st Round': 1, '2nd Round': 2, '3rd Round': 3, '4th Round': 4,
        'Quarterfinals': 5, 'Semifinals': 6, 'The Final': 7, 'Round Robin': 5
    }
    round_num = round_map.get(round_name, 3)
    
    # Series encoding
    series_map = {
        'Grand Slam': 7, 'Masters Cup': 6, 'Masters 1000': 5,
        'ATP500': 4, 'ATP250': 3, 'International': 2, 'International Gold': 2
    }
    series_num = series_map.get(series, 3)
    
    # Build feature dict matching FEATURE_COLS_V2b order
    features = {
        'elo_diff': elo_p1_global - elo_p2_global,
        'surf_elo_diff': elo_p1_surf - elo_p2_surf,
        'surf_elo_p1': elo_p1_surf,
        'surf_elo_p2': elo_p2_surf,
        'p1_surf_wr_3m': surf_wr_3m_p1,
        'p2_surf_wr_3m': surf_wr_3m_p2,
        'p1_surf_wr_12m': surf_wr_12m_p1,
        'p2_surf_wr_12m': surf_wr_12m_p2,
        'p1_form_5': form_5_p1,
        'p2_form_5': form_5_p2,
        'p1_form_10': form_10_p1,
        'p2_form_10': form_10_p2,
        'h2h_p1_wr': h2h_wr,
        'h2h_total': h2h_total,
        'p1_fatigue': fatigue_p1,
        'p2_fatigue': fatigue_p2,
        'rank_diff': rank_diff,
        'rank_ratio': rank_ratio,
        'pts_diff': pts_diff,
        'is_hard': is_hard,
        'is_clay': is_clay,
        'is_grass': is_grass,
        'best_of_5': best_of_5,
        'round_num': round_num,
        'series_num': series_num,
    }
    
    # Order according to config
    feature_cols = config['feature_cols']
    vector = [features.get(col, 0) for col in feature_cols]
    
    return np.array(vector).reshape(1, -1), features

# ============================================================================
# CALCUL DE MISE (KELLY + FLAT)
# ============================================================================

BETTING_STRATEGIES = {
    "🛡️ PLATE PRUDENTE": {
        "type": "flat",
        "stake_pct": 0.02,
        "min_edge": 0.0,
        "description": "Mise plate 2% du bankroll sur chaque pari recommandé par la stratégie"
    },
    "📊 PLATE STANDARD": {
        "type": "flat",
        "stake_pct": 0.03,
        "min_edge": 0.0,
        "description": "Mise plate 3% du bankroll — Recommandée pour cette stratégie"
    },
    "🔥 PLATE AGRESSIVE": {
        "type": "flat",
        "stake_pct": 0.05,
        "min_edge": 0.0,
        "description": "Mise plate 5% du bankroll — Pour bankrolls solides uniquement"
    },
    "💎 KELLY FRACTIONNAIRE": {
        "type": "kelly",
        "kelly_fraction": 6.0,
        "min_edge": 0.0,
        "max_bet_fraction": 0.08,
        "min_bet_pct": 0.01,
        "description": "Kelly / 6 — Mise proportionnelle à l'edge, max 8% du bankroll"
    },
}

def calculate_stake(proba_model, odds, bankroll, strategy):
    """Calcule la mise selon la stratégie choisie"""
    p_market = 1.0 / odds if odds > 0 else 0
    edge = proba_model - p_market
    ev = (proba_model * odds) - 1
    
    if strategy['type'] == 'flat':
        stake_pct = strategy['stake_pct']
        stake = bankroll * stake_pct
        return {
            'stake': stake,
            'edge': edge,
            'ev': ev,
            'should_bet': True,
            'stake_pct': stake_pct,
            'reason': 'OK'
        }
    
    # Kelly
    kelly_fraction = strategy['kelly_fraction']
    max_bet_fraction = strategy['max_bet_fraction']
    min_bet_pct = strategy['min_bet_pct']
    
    q = 1 - proba_model
    b = odds - 1
    if b <= 0:
        return {'stake': 0, 'edge': edge, 'ev': ev, 'should_bet': False, 
                'stake_pct': 0, 'reason': 'Cote invalide'}
    
    kelly_raw = (proba_model * b - q) / b
    kelly_adjusted = kelly_raw / kelly_fraction
    stake_pct = max(min_bet_pct, min(kelly_adjusted, max_bet_fraction))
    stake = bankroll * stake_pct
    
    return {
        'stake': stake,
        'edge': edge,
        'ev': ev,
        'should_bet': ev > 0,
        'stake_pct': stake_pct,
        'kelly_raw': kelly_raw,
        'reason': 'OK' if ev > 0 else f'EV {ev:.1%} <= 0'
    }

# ============================================================================
# GESTION DES PARIS (FICHIER LOCAL)
# ============================================================================

def get_bets_file():
    return BETS_DIR / "bets.csv"

def init_bankroll(default=1000.0):
    """Initialise ou charge la bankroll"""
    bankroll_file = BETS_DIR / "bankroll.json"
    if bankroll_file.exists():
        with open(bankroll_file, 'r') as f:
            data = json.load(f)
            return data.get('bankroll', default)
    else:
        save_bankroll(default)
        return default

def save_bankroll(amount):
    """Sauvegarde la bankroll"""
    bankroll_file = BETS_DIR / "bankroll.json"
    with open(bankroll_file, 'w') as f:
        json.dump({'bankroll': amount, 'updated': datetime.now().isoformat()}, f)

def add_bet(tournament, round_name, player1, player2, pick, odds, stake, 
            model_prob, edge, ev):
    """Ajoute un pari au fichier CSV"""
    bets_file = get_bets_file()
    
    new_bet = pd.DataFrame([{
        'bet_id': int(time.time()),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'tournament': tournament,
        'round': round_name,
        'player_1': player1,
        'player_2': player2,
        'pick': pick,
        'odds': odds,
        'stake': stake,
        'model_prob': model_prob,
        'edge': edge,
        'ev': ev,
        'status': 'open',
        'result': '',
        'profit': 0.0
    }])
    
    if bets_file.exists():
        existing = pd.read_csv(bets_file)
        df = pd.concat([existing, new_bet], ignore_index=True)
    else:
        df = new_bet
    
    df.to_csv(bets_file, index=False)
    return True

def get_open_bets():
    """Retourne les paris ouverts"""
    bets_file = get_bets_file()
    if not bets_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(bets_file)
    return df[df['status'] == 'open']

def close_bet(bet_id, result):
    """Ferme un pari avec un résultat"""
    bets_file = get_bets_file()
    if not bets_file.exists():
        return False
    
    df = pd.read_csv(bets_file)
    mask = df['bet_id'] == bet_id
    
    if not mask.any():
        return False
    
    df.loc[mask, 'status'] = 'closed'
    df.loc[mask, 'result'] = result
    
    if result == 'win':
        df.loc[mask, 'profit'] = df.loc[mask, 'stake'] * (df.loc[mask, 'odds'] - 1)
    elif result == 'loss':
        df.loc[mask, 'profit'] = -df.loc[mask, 'stake']
    else:  # void
        df.loc[mask, 'profit'] = 0
    
    df.to_csv(bets_file, index=False)
    return True

def get_all_bets():
    """Retourne tous les paris"""
    bets_file = get_bets_file()
    if not bets_file.exists():
        return pd.DataFrame()
    return pd.read_csv(bets_file)

# ============================================================================
# PRÉDICTION
# ============================================================================

def predict_match(player1, player2, surface, series, round_name, best_of,
                  rank1, rank2, pts1, pts2, odds1, odds2):
    """Prédit le résultat d'un match et identifie les value bets"""
    
    model, scaler = load_model()
    elo_data = load_elo_ratings()
    config = load_model_config()
    recent_matches = load_recent_matches()
    
    # Build features
    X, features = build_feature_vector(
        player1, player2, surface, series, round_name, best_of,
        rank1, rank2, pts1, pts2, elo_data, recent_matches, config
    )
    
    # Scale + predict
    X_scaled = scaler.transform(X)
    proba_p1 = model.predict_proba(X_scaled)[0, 1]
    proba_p2 = 1 - proba_p1
    
    # Market probabilities
    fair_prob_1 = (1/odds1) / (1/odds1 + 1/odds2) if odds1 > 0 and odds2 > 0 else 0.5
    fair_prob_2 = 1 - fair_prob_1
    margin = (1/odds1 + 1/odds2 - 1) * 100 if odds1 > 0 and odds2 > 0 else 0
    
    # Edges
    edge_p1 = proba_p1 - fair_prob_1
    edge_p2 = proba_p2 - fair_prob_2
    ev_p1 = proba_p1 * odds1 - 1
    ev_p2 = proba_p2 * odds2 - 1
    
    # Strategy check
    strategy = config['strategy']
    is_eligible = (
        series in strategy['series_filter'] and
        round_name in strategy['rounds_filter']
    )
    
    # Determine best bet
    best_bet = None
    if is_eligible:
        if proba_p1 > strategy['model_threshold']:
            best_bet = {
                'player': player1,
                'opponent': player2,
                'proba': proba_p1,
                'odds': odds1,
                'edge': edge_p1,
                'ev': ev_p1,
                'side': 'P1'
            }
        if proba_p2 > strategy['model_threshold']:
            if best_bet is None or proba_p2 > best_bet['proba']:
                best_bet = {
                    'player': player2,
                    'opponent': player1,
                    'proba': proba_p2,
                    'odds': odds2,
                    'edge': edge_p2,
                    'ev': ev_p2,
                    'side': 'P2'
                }
    
    return {
        'proba_p1': proba_p1,
        'proba_p2': proba_p2,
        'fair_prob_1': fair_prob_1,
        'fair_prob_2': fair_prob_2,
        'margin': margin,
        'edge_p1': edge_p1,
        'edge_p2': edge_p2,
        'ev_p1': ev_p1,
        'ev_p2': ev_p2,
        'is_eligible': is_eligible,
        'best_bet': best_bet,
        'features': features,
        'elo_p1_global': features['elo_diff'] / 2 + 1500,  # approx
        'elo_p2_global': 1500 - features['elo_diff'] / 2,
        'elo_p1_surf': features['surf_elo_p1'],
        'elo_p2_surf': features['surf_elo_p2'],
    }

# ============================================================================
# PAGE D'ACCUEIL
# ============================================================================

def show_home_page():
    """Page d'accueil avec statistiques du modèle"""
    
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1>🎾 ATP Tennis Value Betting 🎾</h1>
        <p style="font-size: 1.2rem; color: #888;">
            Modèle XGBoost + Elo par surface — Stratégie Grand Slam validée
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Performance du Modèle (Backtest 2020-2025)")
    
    cols = st.columns(4)
    metrics = [
        ("+6.5%", "ROI"),
        ("84%", "Win Rate"),
        ("5/6", "Années rentables"),
        ("1.36", "Sharpe Ratio"),
    ]
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Stratégie : Grand Slam Late Rounds")
    
    st.markdown("""
    <div class="card">
        <h4>📋 Règles de la stratégie</h4>
        <ol style="line-height: 2;">
            <li><b>Filtre tournoi</b> : Uniquement les <b>Grand Slams</b> (Australian Open, Roland-Garros, Wimbledon, US Open)</li>
            <li><b>Filtre round</b> : Uniquement les <b>Quarts de finale, Demi-finales et Finales</b></li>
            <li><b>Filtre modèle</b> : Parier sur le joueur dont la probabilité modèle est <b>> 60%</b></li>
            <li><b>Mise plate</b> : 2-3% du bankroll par pari (pas de Kelly, volume trop faible)</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #4CAF50;">✅ Pourquoi ça marche</h4>
            <ul style="line-height: 1.8;">
                <li>Les meilleurs joueurs dominent en GS (Best of 5)</li>
                <li>Le modèle Elo par surface capture les spécialistes</li>
                <li>Bookmakers sous-estiment légèrement les gros favoris en late rounds GS</li>
                <li>Format long = moins de variance = modèle plus fiable</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #FFC107;">⚠️ Limites</h4>
            <ul style="line-height: 1.8;">
                <li>~23 paris/an (faible volume)</li>
                <li>p-value = 0.088 (significatif à 10%, pas à 5%)</li>
                <li>Max drawdown: -4.3 unités</li>
                <li>2024 seule année perdante (-4%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Variantes
    st.markdown("### 🔀 Variantes de la stratégie")
    
    var_cols = st.columns(3)
    variants = [
        ("A: Standard", "> 60%", "QF/SF/F", "+6.5%", "116", "84%"),
        ("B: Prudente", "> 80%", "QF/SF/F", "+9.4%", "57", "93%"),
        ("C: SF Only", "> 55%", "SF only", "+11.0%", "35", "86%"),
    ]
    for col, (name, thresh, rounds, roi, n, wr) in zip(var_cols, variants):
        with col:
            st.markdown(f"""
            <div class="card">
                <h5>{name}</h5>
                <p>Seuil: {thresh}<br>Rounds: {rounds}<br>
                <b>ROI: {roi}</b> | N={n} | Win: {wr}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE DE PRÉDICTION
# ============================================================================

def show_prediction_page():
    """Page de prédiction de match"""
    
    st.title("🎾 Prédiction de Match")
    
    config = load_model_config()
    player_stats = load_player_stats()
    
    # Liste des joueurs pour l'autocomplétion
    all_players = sorted(player_stats.keys())
    
    st.markdown("""
    <div class="card">
        <h4>📋 Saisie du match</h4>
        <p>Entrez les détails du match Grand Slam à analyser</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Match details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Joueur 1")
        player1 = st.selectbox("Joueur 1", all_players, index=None, 
                               placeholder="Tapez un nom...", key="p1")
        
        rank1 = st.number_input("Classement ATP", min_value=1, max_value=2000, 
                                value=10, key="rank1")
        pts1 = st.number_input("Points ATP", min_value=0, max_value=20000, 
                               value=3000, key="pts1")
        odds1 = st.number_input("Cote Joueur 1", min_value=1.01, max_value=50.0, 
                                value=1.50, step=0.05, key="odds1")
    
    with col2:
        st.markdown("#### 🔵 Joueur 2")
        player2 = st.selectbox("Joueur 2", all_players, index=None, 
                               placeholder="Tapez un nom...", key="p2")
        
        rank2 = st.number_input("Classement ATP", min_value=1, max_value=2000, 
                                value=20, key="rank2")
        pts2 = st.number_input("Points ATP", min_value=0, max_value=20000, 
                               value=1500, key="pts2")
        odds2 = st.number_input("Cote Joueur 2", min_value=1.01, max_value=50.0, 
                                value=2.50, step=0.05, key="odds2")
    
    # Context
    ctx_cols = st.columns(4)
    with ctx_cols[0]:
        tournament = st.selectbox("Tournoi", [
            "Australian Open", "French Open", "Wimbledon", "US Open"
        ])
    with ctx_cols[1]:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    with ctx_cols[2]:
        round_name = st.selectbox("Round", [
            "Quarterfinals", "Semifinals", "The Final"
        ])
    with ctx_cols[3]:
        best_of = st.selectbox("Format", [5, 3], index=0)
    
    # Auto-detect surface
    surface_map = {"Australian Open": "Hard", "French Open": "Clay", 
                   "Wimbledon": "Grass", "US Open": "Hard"}
    surface = surface_map.get(tournament, surface)
    
    st.markdown("---")
    
    # Predict button
    if st.button("🎯 Analyser le match", type="primary", use_container_width=True):
        if not player1 or not player2:
            st.error("❌ Veuillez sélectionner les deux joueurs")
            return
        
        if player1 == player2:
            st.error("❌ Les deux joueurs doivent être différents")
            return
        
        with st.spinner("Analyse en cours..."):
            prediction = predict_match(
                player1, player2, surface, "Grand Slam", round_name, best_of,
                rank1, rank2, pts1, pts2, odds1, odds2
            )
        
        # Store prediction in session
        st.session_state.last_prediction = prediction
        st.session_state.last_match = {
            'player1': player1, 'player2': player2, 'surface': surface,
            'tournament': tournament, 'round': round_name,
            'odds1': odds1, 'odds2': odds2
        }
        
        # Display results
        display_prediction(prediction, player1, player2, surface, tournament, 
                          round_name, odds1, odds2)

def get_surface_emoji(surface):
    if surface == "Clay":
        return "🧱"
    elif surface == "Grass":
        return "🌿"
    else:
        return "🏟️"

def display_prediction(pred, p1, p2, surface, tournament, round_name, odds1, odds2):
    """Affiche les résultats de la prédiction"""
    
    surf_emoji = get_surface_emoji(surface)
    
    st.markdown(f"### {surf_emoji} {tournament} — {round_name}")
    st.markdown(f"**{p1}** vs **{p2}** | Surface: **{surface}** | Marge bookmaker: {pred['margin']:.1f}%")
    
    # Tableau de comparaison
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <h3>🟢 {p1}</h3>
            <div class="metric-value" style="color: {'#4CAF50' if pred['proba_p1'] > pred['proba_p2'] else '#F44336'};">
                {pred['proba_p1']:.1%}
            </div>
            <p>Elo Surface: {pred['elo_p1_surf']:.0f}</p>
            <p>Cote: <b>{odds1:.2f}</b></p>
            <p>Marché: {pred['fair_prob_1']:.1%}</p>
            <p style="color: {'#4CAF50' if pred['edge_p1'] > 0 else '#F44336'};">
                Edge: {pred['edge_p1']*100:+.1f}%
            </p>
            <p style="color: {'#4CAF50' if pred['ev_p1'] > 0 else '#F44336'};">
                EV: {pred['ev_p1']*100:+.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding-top: 60px;">
            <h2>VS</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <h3>🔵 {p2}</h3>
            <div class="metric-value" style="color: {'#4CAF50' if pred['proba_p2'] > pred['proba_p1'] else '#F44336'};">
                {pred['proba_p2']:.1%}
            </div>
            <p>Elo Surface: {pred['elo_p2_surf']:.0f}</p>
            <p>Cote: <b>{odds2:.2f}</b></p>
            <p>Marché: {pred['fair_prob_2']:.1%}</p>
            <p style="color: {'#4CAF50' if pred['edge_p2'] > 0 else '#F44336'};">
                Edge: {pred['edge_p2']*100:+.1f}%
            </p>
            <p style="color: {'#4CAF50' if pred['ev_p2'] > 0 else '#F44336'};">
                EV: {pred['ev_p2']*100:+.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendation
    st.markdown("---")
    
    if pred['best_bet']:
        bet = pred['best_bet']
        st.markdown(f"""
        <div class="bet-recommendation">
            <h4>✅ RECOMMANDATION DE PARI</h4>
            <p><b>Parier sur :</b> 🎾 <b>{bet['player']}</b></p>
            <p><b>Cote :</b> {bet['odds']:.2f}</p>
            <p><b>Probabilité modèle :</b> {bet['proba']:.1%}</p>
            <p><b>Edge :</b> {bet['edge']*100:+.1f}%</p>
            <p><b>EV :</b> {bet['ev']*100:+.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stake calculation
        current_bankroll = init_bankroll()
        strategy_name = st.session_state.get('selected_strategy', "📊 PLATE STANDARD")
        strategy = BETTING_STRATEGIES.get(strategy_name, BETTING_STRATEGIES["📊 PLATE STANDARD"])
        
        stake_info = calculate_stake(bet['proba'], bet['odds'], current_bankroll, strategy)
        
        st.markdown(f"""
        <div class="card">
            <h5>💰 Calcul de mise ({strategy_name})</h5>
            <p><b>Bankroll :</b> {current_bankroll:.2f} €</p>
            <p><b>Mise recommandée :</b> {stake_info['stake']:.2f} € ({stake_info['stake_pct']:.1%} du bankroll)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Save bet button
        match_info = st.session_state.get('last_match', {})
        if st.button("💾 Enregistrer ce pari", type="primary"):
            success = add_bet(
                tournament=match_info.get('tournament', ''),
                round_name=match_info.get('round', ''),
                player1=match_info.get('player1', ''),
                player2=match_info.get('player2', ''),
                pick=bet['player'],
                odds=bet['odds'],
                stake=stake_info['stake'],
                model_prob=bet['proba'],
                edge=bet['edge'],
                ev=bet['ev']
            )
            if success:
                # Update bankroll
                new_bankroll = current_bankroll - stake_info['stake']
                save_bankroll(new_bankroll)
                st.success(f"✅ Pari enregistré : {stake_info['stake']:.2f}€ sur {bet['player']} @ {bet['odds']:.2f}")
                st.rerun()
    else:
        if not pred['is_eligible']:
            st.markdown("""
            <div class="no-bet">
                <h4>ℹ️ Match hors stratégie</h4>
                <p>Ce match n'est pas dans le périmètre de la stratégie (Grand Slam QF/SF/F uniquement).</p>
                <p>L'analyse est affichée à titre informatif.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-bet">
                <h4>ℹ️ Aucun pari recommandé</h4>
                <p>Aucun joueur n'atteint le seuil de probabilité modèle (> 60%).</p>
                <p>{p1}: {pred['proba_p1']:.1%} | {p2}: {pred['proba_p2']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature details
    with st.expander("🔍 Détails des features"):
        features = pred['features']
        feat_df = pd.DataFrame([
            {"Feature": k, "Valeur": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in features.items()
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE ÉVÉNEMENTS (LIVE ODDS)
# ============================================================================

def show_events_page():
    """Page des événements ATP avec cotes en direct"""
    
    st.title("📡 Événements ATP — Cotes en Direct")
    
    config = load_model_config()
    player_stats = load_player_stats()
    elo_data = load_elo_ratings()
    recent_matches = load_recent_matches()
    all_players = list(player_stats.keys())
    
    # API key status
    api_key = get_odds_api_key()
    
    if not api_key:
        st.error("❌ Aucune clé API configurée")
        st.markdown("""
        <div class="card">
            <h4>🔑 Configuration de l'API</h4>
            <p>Obtenez une clé gratuite sur <a href="https://the-odds-api.com/#get-access">The Odds API</a> (500 req/mois)</p>
        </div>
        """, unsafe_allow_html=True)
        temp_key = st.text_input("Clé API (temporaire)", type="password", key="temp_api_input")
        if temp_key:
            st.session_state.temp_odds_api_key = temp_key
            st.rerun()
        return
    
    # Bouton de rafraîchissement
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        refresh = st.button("🔄 Rafraîchir les cotes", type="primary")
    with col_info:
        st.caption("Source: The Odds API | 500 req/mois gratuites")
    
    if refresh or 'events_data' not in st.session_state:
        with st.spinner("🔍 Recherche des tournois actifs..."):
            sports, msg_sports = fetch_active_tennis_sports(api_key)
            st.session_state.sports_msg = msg_sports
            
            if not sports:
                st.warning(f"Aucun tournoi ATP actif en ce moment. {msg_sports}")
                st.session_state.events_data = {}
                st.session_state.events_info = {}
            else:
                all_events = {}
                all_info = {}
                for sport in sports:
                    sport_key = sport['key']
                    tournament_info = infer_tournament_info(sport_key, sport.get('title', ''))
                    with st.spinner(f"📡 {tournament_info.get('name', sport_key)}..."):
                        events, msg = fetch_tennis_odds(sport_key, api_key)
                        if events:
                            all_events[sport_key] = events
                            all_info[sport_key] = {
                                'msg': msg,
                                'info': tournament_info,
                                'title': sport.get('title', sport_key)
                            }
                
                st.session_state.events_data = all_events
                st.session_state.events_info = all_info
    
    # Display status
    st.caption(st.session_state.get('sports_msg', ''))
    
    events_data = st.session_state.get('events_data', {})
    events_info = st.session_state.get('events_info', {})
    
    if not events_data:
        st.info("📭 Aucun événement ATP en cours. Cliquez sur 🔄 pour vérifier.")
        return
    
    # Bookmaker filter
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 Paramètres cotes")
    show_all_books = st.sidebar.checkbox("Afficher tous les bookmakers", value=False)
    show_non_gs = st.sidebar.checkbox("Afficher les tournois non-GS", value=True)
    
    # Display each tournament
    for sport_key, events in events_data.items():
        info = events_info.get(sport_key, {})
        tournament_info = info.get('info', {})
        # Fallback: si info vide, inférer
        if not tournament_info:
            tournament_info = infer_tournament_info(sport_key, info.get('title', sport_key))
        tournament_name = tournament_info.get('name', sport_key)
        series = tournament_info.get('series', 'ATP')
        surface = tournament_info.get('surface', 'Hard')
        best_of = tournament_info.get('best_of', 3)
        is_gs = series == 'Grand Slam'
        
        if not show_non_gs and not is_gs:
            continue
        
        surf_emoji = get_surface_emoji(surface)
        gs_badge = ' 🏆' if is_gs else ''
        
        st.markdown(f"### {surf_emoji} {tournament_name}{gs_badge}")
        st.caption(f"{series} | Surface: {surface} | Best of {best_of} | {info.get('msg', '')}")
        
        for event in events:
            home = event['home']
            away = event['away']
            odds_h = event.get('odds_home')
            odds_a = event.get('odds_away')
            bookmaker = event.get('bookmaker', 'N/A')
            commence = event.get('commence_time', '')
            
            # Parse time
            try:
                match_time = datetime.fromisoformat(commence.replace('Z', '+00:00'))
                time_str = match_time.strftime('%d/%m %H:%M')
            except Exception:
                time_str = commence[:16] if commence else '?'
            
            # Match API names to model names
            model_home = match_api_name_to_model(home, all_players)
            model_away = match_api_name_to_model(away, all_players)
            
            matched_home = model_home or home
            matched_away = model_away or away
            match_status = '✅' if (model_home and model_away) else '⚠️'
            
            with st.container():
                st.markdown(f"---")
                
                # Match header
                head_cols = st.columns([1, 3, 3, 2])
                with head_cols[0]:
                    st.markdown(f"**⏰ {time_str}**")
                with head_cols[1]:
                    fav_h = "⭐" if odds_h and odds_a and odds_h < odds_a else ""
                    st.markdown(f"{fav_h} **{home}**")
                    if model_home:
                        elo_g, elo_s = get_player_elo(model_home, elo_data, surface)
                        st.caption(f"Elo: {elo_g:.0f} | {surface}: {elo_s:.0f}")
                    else:
                        st.caption("⚠️ Joueur non trouvé dans le modèle")
                with head_cols[2]:
                    fav_a = "⭐" if odds_h and odds_a and odds_a < odds_h else ""
                    st.markdown(f"{fav_a} **{away}**")
                    if model_away:
                        elo_g, elo_s = get_player_elo(model_away, elo_data, surface)
                        st.caption(f"Elo: {elo_g:.0f} | {surface}: {elo_s:.0f}")
                    else:
                        st.caption("⚠️ Joueur non trouvé dans le modèle")
                with head_cols[3]:
                    if odds_h and odds_a:
                        st.markdown(f"📚 **{bookmaker}**")
                        st.caption(f"{odds_h:.2f} — {odds_a:.2f}")
                    else:
                        st.caption("Pas de cotes")
                
                # Prediction if both players matched
                if model_home and model_away and odds_h and odds_a:
                    # Get latest rankings from recent_matches
                    rank_h, pts_h = get_latest_rank(model_home, recent_matches)
                    rank_a, pts_a = get_latest_rank(model_away, recent_matches)
                    
                    # Use a reasonable round for unknown
                    round_name = 'Quarterfinals'  # Default — user can override
                    
                    pred = predict_match(
                        model_home, model_away, surface, series, round_name, best_of,
                        rank_h, rank_a, pts_h, pts_a, odds_h, odds_a
                    )
                    
                    # Show prediction bar
                    pred_cols = st.columns([2, 1, 2])
                    with pred_cols[0]:
                        color_h = '#4CAF50' if pred['proba_p1'] > 0.5 else '#F44336'
                        st.markdown(f"<span style='color:{color_h}; font-size:1.3rem;'><b>{pred['proba_p1']:.1%}</b></span>"
                                    f" | Edge: <span style='color:{color_h};'>{pred['edge_p1']*100:+.1f}%</span>",
                                    unsafe_allow_html=True)
                    with pred_cols[1]:
                        margin = pred['margin']
                        st.caption(f"Marge: {margin:.1f}%")
                    with pred_cols[2]:
                        color_a = '#4CAF50' if pred['proba_p2'] > 0.5 else '#F44336'
                        st.markdown(f"<span style='color:{color_a}; font-size:1.3rem;'><b>{pred['proba_p2']:.1%}</b></span>"
                                    f" | Edge: <span style='color:{color_a};'>{pred['edge_p2']*100:+.1f}%</span>",
                                    unsafe_allow_html=True)
                    
                    # Strategy eligibility
                    strategy = config['strategy']
                    
                    # Allow user to specify round for this match
                    with st.expander(f"⚙️ Réglages pour {home} vs {away}"):
                        round_override = st.selectbox(
                            "Round réel",
                            ['1st Round', '2nd Round', '3rd Round', '4th Round',
                             'Quarterfinals', 'Semifinals', 'The Final'],
                            index=4,
                            key=f"round_{event.get('event_id', '')}"
                        )
                        
                        if round_override != round_name:
                            pred = predict_match(
                                model_home, model_away, surface, series, round_override, best_of,
                                rank_h, rank_a, pts_h, pts_a, odds_h, odds_a
                            )
                        
                        is_eligible = (
                            series in strategy['series_filter'] and
                            round_override in strategy['rounds_filter']
                        )
                        
                        if show_all_books:
                            st.markdown("**📚 Toutes les cotes :**")
                            for bk in event.get('all_bookmakers', []):
                                odds_str = " | ".join([f"{n}: {o:.2f}" for n, o in bk['odds'].items()])
                                st.caption(f"{bk['name']:20s} → {odds_str}")
                    
                    # Value bet badge
                    if pred['best_bet']:
                        bet = pred['best_bet']
                        is_eligible_check = (
                            series in strategy['series_filter'] and
                            round_name in strategy['rounds_filter']
                        )
                        
                        if is_eligible_check:
                            current_bankroll = init_bankroll()
                            strategy_name = st.session_state.get('selected_strategy', "📊 PLATE STANDARD")
                            strat = BETTING_STRATEGIES.get(strategy_name, BETTING_STRATEGIES["📊 PLATE STANDARD"])
                            stake_info = calculate_stake(bet['proba'], bet['odds'], current_bankroll, strat)
                            
                            st.markdown(f"""
                            <div class="bet-recommendation">
                                <b>✅ VALUE BET</b> — Parier sur <b>{bet['player']}</b> 
                                @ {bet['odds']:.2f} | Prob: {bet['proba']:.1%} 
                                | Edge: {bet['edge']*100:+.1f}% | EV: {bet['ev']*100:+.1f}%
                                | Mise: {stake_info['stake']:.2f}€ ({stake_info['stake_pct']:.1%})
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button(f"💾 Enregistrer le pari sur {bet['player']}", key=f"bet_{event.get('event_id', '')}"):
                                success = add_bet(
                                    tournament=tournament_name,
                                    round_name=round_name,
                                    player1=home,
                                    player2=away,
                                    pick=bet['player'],
                                    odds=bet['odds'],
                                    stake=stake_info['stake'],
                                    model_prob=bet['proba'],
                                    edge=bet['edge'],
                                    ev=bet['ev']
                                )
                                if success:
                                    save_bankroll(current_bankroll - stake_info['stake'])
                                    st.success(f"✅ Pari enregistré : {stake_info['stake']:.2f}€ sur {bet['player']}")
                                    st.rerun()
                        else:
                            st.markdown(f"""
                            <div class="no-bet">
                                <b>📊 Signal modèle</b> — {bet['player']} ({bet['proba']:.1%})
                                mais <b>hors stratégie</b> ({series} {round_name})
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    if not is_gs:
                        pass  # Don't show anything for non-GS with no signal
                    else:
                        st.caption(f"Aucun value bet détecté (P1: {pred['proba_p1']:.1%}, P2: {pred['proba_p2']:.1%})")

# ============================================================================
# PAGE BANKROLL
# ============================================================================

def show_bankroll_page():
    """Gestion de la bankroll et historique des paris"""
    
    st.title("💰 Gestion de la Bankroll")
    
    current_bankroll = init_bankroll()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💵 Bankroll actuelle", f"{current_bankroll:.2f} €")
    
    # Adjust bankroll
    st.markdown("### ⚙️ Ajuster la bankroll")
    adj_cols = st.columns([2, 1, 1])
    with adj_cols[0]:
        adjustment = st.number_input("Montant (€)", min_value=-current_bankroll,
                                     max_value=10000.0, value=0.0, step=10.0)
    with adj_cols[1]:
        action = st.selectbox("Action", ["Dépôt", "Retrait"])
    with adj_cols[2]:
        if st.button("✅ Valider", type="primary"):
            if adjustment != 0:
                amt = abs(adjustment) if action == "Dépôt" else -abs(adjustment)
                new_bankroll = current_bankroll + amt
                if new_bankroll < 0:
                    st.error("❌ Bankroll ne peut pas être négative")
                else:
                    save_bankroll(new_bankroll)
                    st.success(f"✅ Bankroll mise à jour : {new_bankroll:.2f}€")
                    st.rerun()
    
    # Open bets
    st.markdown("---")
    st.markdown("### 📋 Paris en cours")
    
    open_bets = get_open_bets()
    
    if not open_bets.empty:
        total_stake = open_bets['stake'].sum()
        st.metric("📊 Paris ouverts", f"{len(open_bets)} | Mise totale: {total_stake:.2f}€")
        
        for idx, bet in open_bets.iterrows():
            with st.expander(f"🎾 {bet['pick']} @ {bet['odds']:.2f} — {bet['tournament']} {bet['round']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Match:** {bet['player_1']} vs {bet['player_2']}")
                    st.write(f"**Pari sur:** {bet['pick']}")
                    st.write(f"**Cote:** {bet['odds']:.2f}")
                    st.write(f"**Mise:** {bet['stake']:.2f} €")
                with col2:
                    st.write(f"**Date:** {bet['date']}")
                    st.write(f"**Prob modèle:** {bet['model_prob']:.1%}")
                    st.write(f"**Edge:** {bet['edge']:.1%}")
                    st.write(f"**EV:** {bet['ev']:.1%}")
                
                st.markdown("**Résultat :**")
                res_cols = st.columns(3)
                with res_cols[0]:
                    if st.button("✅ Victoire", key=f"win_{bet['bet_id']}"):
                        if close_bet(int(bet['bet_id']), "win"):
                            profit = bet['stake'] * (bet['odds'] - 1)
                            save_bankroll(current_bankroll + profit)
                            st.success(f"✅ +{profit:.2f}€")
                            st.rerun()
                with res_cols[1]:
                    if st.button("❌ Défaite", key=f"loss_{bet['bet_id']}"):
                        if close_bet(int(bet['bet_id']), "loss"):
                            save_bankroll(current_bankroll - bet['stake'])
                            st.warning(f"❌ -{bet['stake']:.2f}€")
                            st.rerun()
                with res_cols[2]:
                    if st.button("⚪ Annulé", key=f"void_{bet['bet_id']}"):
                        if close_bet(int(bet['bet_id']), "void"):
                            save_bankroll(current_bankroll + bet['stake'])
                            st.info("⚪ Pari annulé, mise remboursée")
                            st.rerun()
    else:
        st.info("📭 Aucun pari en cours")
    
    # History
    st.markdown("---")
    st.markdown("### 📊 Historique des paris")
    
    all_bets = get_all_bets()
    
    if not all_bets.empty:
        closed = all_bets[all_bets['status'] == 'closed']
        
        if not closed.empty:
            total_bets = len(closed)
            wins = len(closed[closed['result'] == 'win'])
            losses = len(closed[closed['result'] == 'loss'])
            win_rate = wins / total_bets if total_bets > 0 else 0
            total_profit = closed['profit'].sum()
            total_staked = closed['stake'].sum()
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            
            stats_cols = st.columns(5)
            with stats_cols[0]:
                st.metric("Total", total_bets)
            with stats_cols[1]:
                st.metric("Victoires", wins)
            with stats_cols[2]:
                st.metric("Défaites", losses)
            with stats_cols[3]:
                st.metric("Win Rate", f"{win_rate:.1%}")
            with stats_cols[4]:
                st.metric("ROI", f"{roi:.1f}%", delta=f"{total_profit:.2f}€")
            
            # Profit chart
            closed_sorted = closed.sort_values('date')
            closed_sorted['cumulative_profit'] = closed_sorted['profit'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(closed_sorted) + 1)),
                y=closed_sorted['cumulative_profit'],
                mode='lines+markers',
                name='Profit cumulé',
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=8)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            fig.update_layout(
                title="📈 Évolution du profit cumulé",
                xaxis_title="Nombre de paris",
                yaxis_title="Profit (€)",
                hovermode='x unified',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            display_df = closed_sorted[
                ['date', 'tournament', 'round', 'pick', 'odds', 'stake', 'result', 'profit']
            ].copy()
            display_df['odds'] = display_df['odds'].apply(lambda x: f"{x:.2f}")
            display_df['stake'] = display_df['stake'].apply(lambda x: f"{x:.2f}€")
            display_df['profit'] = display_df['profit'].apply(lambda x: f"{x:+.2f}€")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("📭 Aucun pari clôturé")
    else:
        st.info("📭 Aucun historique disponible")

# ============================================================================
# PAGE CLASSEMENT ELO
# ============================================================================

def show_rankings_page():
    """Classement Elo des joueurs"""
    
    st.title("🏆 Classement ATP — Ratings Elo")
    
    player_stats = load_player_stats()
    elo_data = load_elo_ratings()
    
    # Tabs for different rankings
    tab_global, tab_hard, tab_clay, tab_grass = st.tabs([
        "🌍 Global", "🏟️ Hard", "🧱 Clay", "🌿 Grass"
    ])
    
    def display_ranking(elo_dict, title, color):
        ranking = sorted(elo_dict.items(), key=lambda x: -x[1])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            search = st.text_input("🔍 Rechercher", "", key=f"search_{title}")
            top_n = st.slider("Top N", 10, 100, 30, 10, key=f"top_{title}")
        
        if search:
            ranking = [(p, e) for p, e in ranking if search.lower() in p.lower()]
        
        ranking = ranking[:top_n]
        
        if ranking:
            names, elos = zip(*ranking)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(names),
                y=list(elos),
                marker=dict(color=list(elos), colorscale='Viridis', showscale=True),
                text=[f"{e:.0f}" for e in elos],
                textposition='outside'
            ))
            fig.update_layout(
                title=f"🏆 {title} — Top {len(ranking)}",
                xaxis_title="Joueur",
                yaxis_title="Rating Elo",
                height=500,
                template='plotly_dark',
                xaxis={'tickangle': -45}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            df_rank = pd.DataFrame(ranking, columns=['Joueur', 'Elo'])
            df_rank.index = range(1, len(df_rank) + 1)
            df_rank.index.name = 'Rang'
            df_rank['Elo'] = df_rank['Elo'].apply(lambda x: f"{x:.0f}")
            st.dataframe(df_rank, use_container_width=True)
    
    with tab_global:
        display_ranking(elo_data['global'], "Elo Global", "#1E88E5")
    
    with tab_hard:
        display_ranking(elo_data['surface'].get('Hard', {}), "Elo Hard Court", "#1565C0")
    
    with tab_clay:
        display_ranking(elo_data['surface'].get('Clay', {}), "Elo Terre Battue", "#C75B12")
    
    with tab_grass:
        display_ranking(elo_data['surface'].get('Grass', {}), "Elo Gazon", "#388E3C")

# ============================================================================
# PAGE STATISTIQUES
# ============================================================================

def show_stats_page():
    """Statistiques du dataset et du modèle"""
    
    st.title("📊 Statistiques & Analyse")
    
    df = load_historical_data()
    config = load_model_config()
    
    st.markdown("### 📈 Vue d'ensemble du dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎾 Matchs total", f"{len(df):,}")
    with col2:
        st.metric("📅 Période", f"{df['Date'].min().year} — {df['Date'].max().year}")
    with col3:
        n_players = len(set(df['Player_1'].unique()) | set(df['Player_2'].unique()))
        st.metric("👤 Joueurs", f"{n_players:,}")
    with col4:
        gs_count = len(df[df['Series'] == 'Grand Slam'])
        st.metric("🏆 Matchs GS", f"{gs_count:,}")
    
    # Distribution par surface
    tab1, tab2, tab3 = st.tabs(["📊 Surfaces", "🏆 Tournois", "📅 Timeline"])
    
    with tab1:
        surf_counts = df['Surface'].value_counts()
        fig = px.pie(values=surf_counts.values, names=surf_counts.index,
                     title="Distribution par surface", 
                     color_discrete_map={'Hard': '#1565C0', 'Clay': '#C75B12', 
                                        'Grass': '#388E3C', 'Carpet': '#795548'})
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        series_counts = df['Series'].value_counts()
        fig = px.bar(x=series_counts.index, y=series_counts.values,
                     title="Distribution par série de tournois",
                     labels={'x': 'Série', 'y': 'Nombre de matchs'})
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        yearly = df.groupby(df['Date'].dt.year).size().reset_index(name='count')
        fig = px.line(yearly, x='Date', y='count', 
                      title="Nombre de matchs par année",
                      labels={'Date': 'Année', 'count': 'Matchs'})
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    # Model features
    st.markdown("### 🧠 Features du modèle V2b")
    st.markdown(f"**{len(config['feature_cols'])} features** utilisées :")
    
    for i, feat in enumerate(config['feature_cols'], 1):
        st.markdown(f"{i}. `{feat}`")

# ============================================================================
# MISE À JOUR DES DONNÉES — tennis-data.co.uk
# ============================================================================

TENNIS_DATA_URL = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"

def download_tennis_data(year):
    """Télécharge les données ATP d'une année depuis tennis-data.co.uk"""
    url = TENNIS_DATA_URL.format(year=year)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        df = pd.read_excel(io.BytesIO(data))
        return df
    except Exception as e:
        st.error(f"❌ Erreur téléchargement {year}: {e}")
        return None


def transform_tennis_data(df_raw):
    """Transforme le format tennis-data.co.uk vers notre format CSV.
    
    tennis-data: Winner/Loser → notre format: Player_1/Player_2/Winner
    On randomise l'ordre P1/P2 pour éviter le biais (sinon Winner = toujours P1)
    """
    rows = []
    for _, row in df_raw.iterrows():
        # Skip incomplete matches
        if row.get('Comment', '') != 'Completed':
            continue
        if pd.isna(row.get('Winner')) or pd.isna(row.get('Loser')):
            continue
        
        winner = str(row['Winner']).strip()
        loser = str(row['Loser']).strip()
        
        # Reconstruct score from set columns
        score_parts = []
        for s in range(1, 6):
            ws, ls = row.get(f'W{s}'), row.get(f'L{s}')
            if pd.notna(ws) and pd.notna(ls):
                score_parts.append(f"{int(ws)}-{int(ls)}")
        score = ' '.join(score_parts) if score_parts else ''
        
        # Randomize order to avoid Winner always being P1
        if np.random.random() > 0.5:
            p1, p2 = winner, loser
            rank1 = row.get('WRank', -1)
            rank2 = row.get('LRank', -1)
            pts1 = row.get('WPts', -1)
            pts2 = row.get('LPts', -1)
            # Odds: PSW/PSL = Pinnacle Winner/Loser odds
            odd1 = row.get('PSW', row.get('B365W', row.get('AvgW', -1)))
            odd2 = row.get('PSL', row.get('B365L', row.get('AvgL', -1)))
        else:
            p1, p2 = loser, winner
            rank1 = row.get('LRank', -1)
            rank2 = row.get('WRank', -1)
            pts1 = row.get('LPts', -1)
            pts2 = row.get('WPts', -1)
            odd1 = row.get('PSL', row.get('B365L', row.get('AvgL', -1)))
            odd2 = row.get('PSW', row.get('B365W', row.get('AvgW', -1)))
            # Invert score
            score_parts_inv = []
            for s in range(1, 6):
                ws, ls = row.get(f'W{s}'), row.get(f'L{s}')
                if pd.notna(ws) and pd.notna(ls):
                    score_parts_inv.append(f"{int(ls)}-{int(ws)}")
            score = ' '.join(score_parts_inv) if score_parts_inv else ''
        
        # Clean values
        rank1 = int(rank1) if pd.notna(rank1) and rank1 != -1 else -1
        rank2 = int(rank2) if pd.notna(rank2) and rank2 != -1 else -1
        pts1 = int(pts1) if pd.notna(pts1) and pts1 != -1 else -1
        pts2 = int(pts2) if pd.notna(pts2) and pts2 != -1 else -1
        odd1 = float(odd1) if pd.notna(odd1) else -1.0
        odd2 = float(odd2) if pd.notna(odd2) else -1.0
        
        rows.append({
            'Tournament': row.get('Tournament', ''),
            'Date': pd.to_datetime(row['Date']).strftime('%Y-%m-%d'),
            'Series': row.get('Series', ''),
            'Court': row.get('Court', 'Outdoor'),
            'Surface': row.get('Surface', ''),
            'Round': row.get('Round', ''),
            'Best of': int(row.get('Best of', 3)),
            'Player_1': p1,
            'Player_2': p2,
            'Winner': winner,
            'Rank_1': rank1,
            'Rank_2': rank2,
            'Pts_1': pts1,
            'Pts_2': pts2,
            'Odd_1': odd1,
            'Odd_2': odd2,
            'Score': score
        })
    
    return pd.DataFrame(rows)


def check_data_freshness():
    """Vérifie l'état des données locales"""
    csv_path = DATA_DIR / "atp_tennis.csv"
    recent_path = MODELS_DIR / "recent_matches.csv"
    
    info = {
        'has_data': False,
        'last_match_date': None,
        'days_old': None,
        'total_matches': 0,
        'total_players': 0,
        'recent_matches_count': 0,
        'message': ''
    }
    
    if not csv_path.exists():
        info['message'] = '📭 Aucune donnée existante.'
        return info
    
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        info['has_data'] = True
        info['total_matches'] = len(df)
        info['total_players'] = len(set(df['Player_1'].unique()) | set(df['Player_2'].unique()))
        info['last_match_date'] = df['Date'].max()
        info['days_old'] = (pd.Timestamp.now() - df['Date'].max()).days
        
        if recent_path.exists():
            recent = pd.read_csv(recent_path)
            info['recent_matches_count'] = len(recent)
        
        days = info['days_old']
        if days <= 7:
            info['message'] = f"✅ Données à jour (dernier match: {info['last_match_date'].date()}, il y a {days} jours)"
        elif days <= 30:
            info['message'] = f"🟡 Données récentes (dernier match: {info['last_match_date'].date()}, il y a {days} jours)"
        else:
            info['message'] = f"🟠 Données à mettre à jour (dernier match: {info['last_match_date'].date()}, il y a {days} jours)"
        
        return info
    except Exception as e:
        info['message'] = f"❌ Erreur lecture: {e}"
        return info


def fetch_new_matches(progress_callback=None):
    """Télécharge les matchs manquants depuis tennis-data.co.uk"""
    csv_path = DATA_DIR / "atp_tennis.csv"
    
    # Load existing data
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_existing['Date'] = pd.to_datetime(df_existing['Date'])
        last_date = df_existing['Date'].max()
    else:
        df_existing = pd.DataFrame()
        last_date = pd.Timestamp('2000-01-01')
    
    current_year = datetime.now().year
    start_year = last_date.year  # Start from the year of last match
    
    all_new = []
    
    for year in range(start_year, current_year + 1):
        if progress_callback:
            progress_callback(f"📥 Téléchargement données {year}...")
        
        df_raw = download_tennis_data(year)
        if df_raw is None:
            continue
        
        if progress_callback:
            progress_callback(f"🔄 Transformation données {year}...")
        
        df_transformed = transform_tennis_data(df_raw)
        if df_transformed.empty:
            continue
        
        df_transformed['Date'] = pd.to_datetime(df_transformed['Date'])
        
        # Keep only matches after our last known date
        new_matches = df_transformed[df_transformed['Date'] > last_date]
        
        if not new_matches.empty:
            all_new.append(new_matches)
            if progress_callback:
                progress_callback(f"✅ {year}: {len(new_matches)} nouveaux matchs trouvés")
        else:
            if progress_callback:
                progress_callback(f"ℹ️ {year}: aucun nouveau match")
    
    if all_new:
        df_new = pd.concat(all_new, ignore_index=True)
        # Deduplicate by Tournament + Date + Player_1 + Player_2 (or Winner)
        df_new = df_new.drop_duplicates(
            subset=['Date', 'Winner', 'Tournament', 'Round'],
            keep='first'
        )
        df_new = df_new.sort_values('Date').reset_index(drop=True)
        return df_new
    
    return pd.DataFrame()


def update_main_csv(df_new):
    """Ajoute les nouveaux matchs au CSV principal"""
    csv_path = DATA_DIR / "atp_tennis.csv"
    
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
    else:
        df_existing = pd.DataFrame()
    
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined['Date'] = pd.to_datetime(df_combined['Date'])
    
    # Deduplicate
    df_combined = df_combined.drop_duplicates(
        subset=['Date', 'Winner', 'Tournament', 'Round'],
        keep='last'
    )
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    
    # Save
    df_combined.to_csv(csv_path, index=False)
    return df_combined


def recalculate_all_elo(progress_callback=None):
    """Recalcule TOUS les Elo (global K=32, surface K=40) sur le dataset complet,
    puis met à jour elo_ratings.pkl, player_stats.pkl, et recent_matches.csv"""
    csv_path = DATA_DIR / "atp_tennis.csv"
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    if progress_callback:
        progress_callback("🧮 Calcul des Elo globaux (K=32)...")
    
    # === Global Elo (K=32) ===
    K_GLOBAL = 32
    INITIAL_ELO = 1500
    elo_global = {}
    
    for _, row in df.iterrows():
        p1, p2, winner = row['Player_1'], row['Player_2'], row['Winner']
        r1 = elo_global.get(p1, INITIAL_ELO)
        r2 = elo_global.get(p2, INITIAL_ELO)
        
        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        s1 = 1.0 if winner == p1 else 0.0
        
        elo_global[p1] = r1 + K_GLOBAL * (s1 - exp1)
        elo_global[p2] = r2 + K_GLOBAL * ((1 - s1) - (1 - exp1))
    
    if progress_callback:
        progress_callback("🎾 Calcul des Elo par surface (K=40)...")
    
    # === Surface Elo (K=40) ===
    K_SURFACE = 40
    surfaces = df['Surface'].unique()
    elo_surface = {s: {} for s in surfaces}
    
    for _, row in df.iterrows():
        surface = row['Surface']
        p1, p2, winner = row['Player_1'], row['Player_2'], row['Winner']
        
        e1 = elo_surface[surface].get(p1, INITIAL_ELO)
        e2 = elo_surface[surface].get(p2, INITIAL_ELO)
        
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        s1 = 1.0 if winner == p1 else 0.0
        
        elo_surface[surface][p1] = e1 + K_SURFACE * (s1 - exp1)
        elo_surface[surface][p2] = e2 + K_SURFACE * ((1 - s1) - (1 - exp1))
    
    if progress_callback:
        progress_callback("💾 Sauvegarde elo_ratings.pkl...")
    
    # === Save elo_ratings.pkl ===
    elo_data = {
        'global': elo_global,
        'surface': {s: dict(v) for s, v in elo_surface.items()}
    }
    joblib.dump(elo_data, MODELS_DIR / "elo_ratings.pkl")
    
    if progress_callback:
        progress_callback("💾 Sauvegarde player_stats.pkl...")
    
    # === Save player_stats.pkl ===
    player_stats = {}
    for player, elo in elo_global.items():
        player_stats[player] = {
            'elo_global': elo,
            'elo_by_surface': {}
        }
        for surface in surfaces:
            if player in elo_surface.get(surface, {}):
                player_stats[player]['elo_by_surface'][surface] = elo_surface[surface][player]
    
    joblib.dump(player_stats, MODELS_DIR / "player_stats.pkl")
    
    if progress_callback:
        progress_callback("💾 Sauvegarde recent_matches.csv...")
    
    # === Save recent_matches.csv (last ~12 months) ===
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
    recent = df[df['Date'] >= cutoff][[
        'Date', 'Tournament', 'Series', 'Surface', 'Round', 'Best of',
        'Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2',
        'Score'
    ]].copy()
    recent.to_csv(MODELS_DIR / "recent_matches.csv", index=False)
    
    return {
        'total_players': len(elo_global),
        'total_matches': len(df),
        'recent_matches': len(recent),
        'top_players': sorted(elo_global.items(), key=lambda x: x[1], reverse=True)[:10]
    }


def show_calendar_page():
    """Page calendrier ATP — tournois en cours, passés et à venir"""
    
    st.title("📅 Calendrier ATP 2025-2026")
    
    df = load_historical_data()
    today = pd.Timestamp.now().normalize()
    
    # Build tournament calendar from data (last 2 seasons)
    cutoff = pd.Timestamp(today.year - 1, 1, 1)
    df_recent = df[df['Date'] >= cutoff].copy()
    
    tourneys = df_recent.groupby(['Tournament', 'Series', 'Surface']).agg(
        start=('Date', 'min'),
        end=('Date', 'max'),
        matches=('Date', 'count'),
    ).reset_index().sort_values('start')
    
    # For tournaments that span 2 years (same name in 2025 and 2026), split them
    # Use last occurrence year for each tournament
    calendar = []
    for _, t in tourneys.iterrows():
        # Get matches for this tournament
        t_matches = df_recent[
            (df_recent['Tournament'] == t['Tournament']) & 
            (df_recent['Series'] == t['Series'])
        ].sort_values('Date')
        
        # Group by year-cluster (if > 60 days gap, it's a different edition)
        dates = t_matches['Date'].values
        editions = []
        current_edition = [dates[0]]
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]) / pd.Timedelta(days=1)
            if gap > 60:
                editions.append(current_edition)
                current_edition = [dates[i]]
            else:
                current_edition.append(dates[i])
        editions.append(current_edition)
        
        for ed_dates in editions:
            ed_start = pd.Timestamp(min(ed_dates))
            ed_end = pd.Timestamp(max(ed_dates))
            n_matches = len(ed_dates)
            
            # Determine status
            if ed_end < today - pd.Timedelta(days=1):
                status = 'past'
            elif ed_start <= today + pd.Timedelta(days=1):
                status = 'live'
            else:
                status = 'upcoming'
            
            calendar.append({
                'tournament': t['Tournament'],
                'series': t['Series'],
                'surface': t['Surface'],
                'start': ed_start,
                'end': ed_end,
                'matches': n_matches,
                'status': status,
                'year': ed_start.year
            })
    
    cal_df = pd.DataFrame(calendar).sort_values('start')
    
    # Also build a static forward calendar for known ATP events (rest of 2026)
    # Based on historical patterns
    atp_2026_schedule = _get_projected_calendar(cal_df, today)
    if not atp_2026_schedule.empty:
        cal_df = pd.concat([cal_df, atp_2026_schedule], ignore_index=True)
        cal_df = cal_df.drop_duplicates(subset=['tournament', 'year', 'start'], keep='first')
        cal_df = cal_df.sort_values('start')
    
    # Surface colors
    surf_colors = {'Hard': '#1565C0', 'Clay': '#C75B12', 'Grass': '#388E3C', 'Carpet': '#795548'}
    surf_emoji = {'Hard': '🔵', 'Clay': '🟤', 'Grass': '🟢', 'Carpet': '⚫'}
    series_emoji = {'Grand Slam': '🏆', 'Masters 1000': '⭐', 'ATP500': '🎾', 'ATP250': '🎪', 'Masters Cup': '👑'}
    status_emoji = {'live': '🔴 EN COURS', 'upcoming': '📅 À venir', 'past': '✅ Terminé'}
    
    # --- LIVE TOURNAMENTS ---
    live = cal_df[cal_df['status'] == 'live']
    if not live.empty:
        st.markdown("### 🔴 Tournois en cours")
        for _, t in live.iterrows():
            se = series_emoji.get(t['series'], '🎾')
            su = surf_emoji.get(t['surface'], '⚪')
            days_left = max(0, (t['end'] - today).days)
            
            st.markdown(f"""
            <div class="card" style="border-left: 4px solid {surf_colors.get(t['surface'], '#888')};">
                <h4>{se} {t['tournament']} {su}</h4>
                <p><b>{t['series']}</b> | Surface: <b>{t['surface']}</b> | 
                {t['start'].strftime('%d/%m')} → {t['end'].strftime('%d/%m/%Y')}</p>
                <p>🔴 <b>En cours</b> — {t['matches']} matchs joués | ~{days_left} jours restants</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📭 Aucun tournoi ATP en cours en ce moment.")
    
    # --- UPCOMING TOURNAMENTS ---
    st.markdown("### 📅 Prochains tournois")
    upcoming = cal_df[(cal_df['status'] == 'upcoming') & (cal_df['start'] >= today)]
    upcoming = upcoming.sort_values('start').head(20)  # Next 20
    
    if not upcoming.empty:
        # Group by month
        upcoming_copy = upcoming.copy()
        upcoming_copy['month'] = upcoming_copy['start'].dt.strftime('%B %Y')
        
        for month, group in upcoming_copy.groupby('month', sort=False):
            st.markdown(f"#### 📆 {month}")
            
            for _, t in group.iterrows():
                se = series_emoji.get(t['series'], '🎾')
                su = surf_emoji.get(t['surface'], '⚪')
                days_until = (t['start'] - today).days
                
                is_gs = t['series'] == 'Grand Slam'
                is_m1000 = 'Masters' in t['series'] or t['series'] == 'Masters Cup'
                highlight = 'font-weight: bold;' if (is_gs or is_m1000) else ''
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    st.markdown(f"<span style='{highlight}'>{se} {t['tournament']}</span>", unsafe_allow_html=True)
                with col2:
                    st.caption(f"{su} {t['surface']} | {t['series']}")
                with col3:
                    st.caption(f"{t['start'].strftime('%d/%m')} → {t['end'].strftime('%d/%m')}")
                with col4:
                    if days_until <= 7:
                        st.caption(f"🔜 J-{days_until}")
                    else:
                        st.caption(f"J-{days_until}")
    else:
        st.info("Aucun prochain tournoi trouvé. Mettez à jour les données dans l'onglet 🔄.")
    
    # --- PAST TOURNAMENTS (collapsible) ---
    st.markdown("---")
    with st.expander("📜 Tournois récents terminés", expanded=False):
        past = cal_df[cal_df['status'] == 'past']
        past = past[past['year'] >= today.year].sort_values('end', ascending=False).head(20)
        
        if not past.empty:
            for _, t in past.iterrows():
                se = series_emoji.get(t['series'], '🎾')
                su = surf_emoji.get(t['surface'], '⚪')
                
                # Find winner of the final
                final_matches = df[
                    (df['Tournament'] == t['tournament']) &
                    (df['Date'] >= t['start']) &
                    (df['Date'] <= t['end']) &
                    (df['Round'].isin(['The Final']))
                ]
                winner = final_matches['Winner'].values[0] if not final_matches.empty else '—'
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                with col1:
                    st.markdown(f"{se} **{t['tournament']}**")
                with col2:
                    st.caption(f"{su} {t['surface']} | {t['series']}")
                with col3:
                    st.caption(f"{t['start'].strftime('%d/%m')} → {t['end'].strftime('%d/%m')}")
                with col4:
                    st.caption(f"🏆 {winner}")
        else:
            st.info("Aucun tournoi passé trouvé pour cette année.")
    
    # --- FULL SEASON VIEW ---
    st.markdown("---")
    st.markdown("### 📊 Vue saison complète")
    
    # Filter for current year
    current_year = today.year
    year_filter = st.radio(
        "Saison", 
        [current_year - 1, current_year],
        index=1,
        horizontal=True
    )
    
    season = cal_df[cal_df['year'] == year_filter].sort_values('start')
    
    if not season.empty:
        # Build a timeline table
        rows = []
        for _, t in season.iterrows():
            se = series_emoji.get(t['series'], '🎾')
            su = surf_emoji.get(t['surface'], '⚪')
            st_emoji = {'live': '🔴', 'upcoming': '📅', 'past': '✅'}.get(t['status'], '')
            rows.append({
                'Statut': st_emoji,
                'Tournoi': f"{se} {t['tournament']}",
                'Série': t['series'],
                'Surface': f"{su} {t['surface']}",
                'Début': t['start'].strftime('%d/%m'),
                'Fin': t['end'].strftime('%d/%m'),
                'Matchs': t['matches']
            })
        
        season_df = pd.DataFrame(rows)
        st.dataframe(season_df, use_container_width=True, hide_index=True, height=600)
    else:
        st.info(f"Pas de données pour la saison {year_filter}.")


def _get_projected_calendar(cal_df, today):
    """Projette les tournois futurs basé sur les patterns de l'année précédente.
    
    Si on a le calendrier 2025, on peut estimer les dates 2026 en ajoutant ~52 semaines.
    """
    projected = []
    last_year = today.year - 1
    
    past_season = cal_df[cal_df['year'] == last_year].copy()
    
    for _, t in past_season.iterrows():
        # Project to current year
        proj_start = t['start'] + pd.DateOffset(years=1)
        proj_end = t['end'] + pd.DateOffset(years=1)
        
        # Only add if it's in the future and not already in cal_df
        if proj_start > today:
            # Check not already present
            existing = cal_df[
                (cal_df['tournament'] == t['tournament']) & 
                (cal_df['year'] == today.year) &
                (abs((cal_df['start'] - proj_start).dt.days) < 30)
            ]
            if existing.empty:
                projected.append({
                    'tournament': t['tournament'],
                    'series': t['series'],
                    'surface': t['surface'],
                    'start': proj_start,
                    'end': proj_end,
                    'matches': 0,  # Projected, no matches yet
                    'status': 'upcoming',
                    'year': proj_start.year
                })
    
    return pd.DataFrame(projected) if projected else pd.DataFrame()


def show_update_page():
    """Page de mise à jour des données — inspirée de l'app UFC"""
    
    st.title("🔄 Mise à jour des données")
    st.markdown("Télécharge les derniers résultats ATP depuis **tennis-data.co.uk** et recalcule tous les Elo.")
    
    # --- État actuel ---
    st.markdown("### 📊 État des données")
    
    freshness = check_data_freshness()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎾 Matchs total", f"{freshness['total_matches']:,}")
    with col2:
        st.metric("👤 Joueurs", f"{freshness['total_players']:,}")
    with col3:
        st.metric("📅 Matchs récents", f"{freshness['recent_matches_count']:,}")
    with col4:
        days = freshness['days_old'] if freshness['days_old'] else '—'
        st.metric("⏰ Âge des données", f"{days} jours")
    
    st.info(freshness['message'])
    
    st.markdown("---")
    
    # --- Boutons d'action ---
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        update_clicked = st.button("🚀 Mettre à jour les données", type="primary", use_container_width=True)
    with col_btn2:
        recalc_clicked = st.button("🧮 Recalculer les Elo", use_container_width=True)
    with col_btn3:
        clear_cache = st.button("🗑️ Vider le cache", use_container_width=True)
    
    if clear_cache:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache vidé avec succès !")
        st.rerun()
    
    # --- Mise à jour complète ---
    if update_clicked:
        st.markdown("### 📥 Mise à jour en cours...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()
        
        logs = []
        def log_progress(msg):
            logs.append(msg)
            status_text.markdown(f"**{msg}**")
            with log_container:
                for l in logs:
                    st.text(l)
        
        try:
            # Step 1: Download new matches
            progress_bar.progress(10)
            log_progress("📡 Recherche de nouveaux matchs...")
            df_new = fetch_new_matches(progress_callback=log_progress)
            
            if df_new.empty:
                progress_bar.progress(100)
                st.success("✅ Les données sont déjà à jour ! Aucun nouveau match trouvé.")
            else:
                # Step 2: Update CSV
                progress_bar.progress(40)
                log_progress(f"💾 Ajout de {len(df_new)} nouveaux matchs au dataset...")
                df_combined = update_main_csv(df_new)
                
                # Step 3: Recalculate Elo
                progress_bar.progress(50)
                log_progress("🧮 Recalcul de tous les Elo...")
                result = recalculate_all_elo(progress_callback=log_progress)
                
                # Step 4: Clear cache
                progress_bar.progress(90)
                log_progress("🗑️ Vidage du cache...")
                st.cache_data.clear()
                st.cache_resource.clear()
                
                progress_bar.progress(100)
                
                # Summary
                st.success(f"""✅ Mise à jour terminée !
                
- **{len(df_new)}** nouveaux matchs ajoutés
- **{result['total_matches']:,}** matchs total dans le dataset
- **{result['total_players']:,}** joueurs avec ratings Elo
- **{result['recent_matches']}** matchs dans recent_matches.csv
                """)
                
                # Show top 10 Elo
                st.markdown("#### 🏆 Top 10 Elo Global après mise à jour")
                top_df = pd.DataFrame(result['top_players'], columns=['Joueur', 'Elo'])
                top_df.index = range(1, len(top_df) + 1)
                top_df['Elo'] = top_df['Elo'].round(1)
                st.dataframe(top_df, use_container_width=True)
                
                st.info("🔄 Rechargez la page pour utiliser les nouvelles données dans les prédictions.")
        
        except Exception as e:
            st.error(f"❌ Erreur pendant la mise à jour: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # --- Recalcul Elo seul ---
    if recalc_clicked:
        st.markdown("### 🧮 Recalcul des Elo en cours...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            progress_bar.progress(20)
            result = recalculate_all_elo(progress_callback=lambda msg: status_text.markdown(f"**{msg}**"))
            
            progress_bar.progress(90)
            st.cache_data.clear()
            st.cache_resource.clear()
            
            progress_bar.progress(100)
            
            st.success(f"""✅ Recalcul terminé !
            
- **{result['total_matches']:,}** matchs traités
- **{result['total_players']:,}** joueurs avec ratings Elo
- **{result['recent_matches']}** matchs récents
            """)
            
            st.markdown("#### 🏆 Top 10 Elo Global")
            top_df = pd.DataFrame(result['top_players'], columns=['Joueur', 'Elo'])
            top_df.index = range(1, len(top_df) + 1)
            top_df['Elo'] = top_df['Elo'].round(1)
            st.dataframe(top_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # --- Info source ---
    st.markdown("---")
    st.markdown("""    
    <div class="card">
        <h4>ℹ️ Source des données</h4>
        <ul>
            <li><b>Source</b> : <a href="http://www.tennis-data.co.uk/alldata.htm" target="_blank">tennis-data.co.uk</a></li>
            <li><b>Format</b> : Fichiers Excel par année (ATP)</li>
            <li><b>Elo Global</b> : K-factor = 32, Elo initial = 1500</li>
            <li><b>Elo Surface</b> : K-factor = 40, Elo initial = 1500</li>
            <li><b>Matchs récents</b> : Derniers 12 mois (pour form, H2H, fatigue)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    
    st.markdown('<div class="main-title">🎾 ATP Tennis Value Betting 🎾</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">XGBoost + Elo Surface — Stratégie Grand Slam Late Rounds</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Bankroll display
        current_bankroll = init_bankroll()
        st.metric("💰 Bankroll", f"{current_bankroll:.2f} €")
        
        st.markdown("---")
        
        # Strategy selection
        st.markdown("### 📊 Stratégie de mise")
        strategy_name = st.selectbox(
            "Choisir la stratégie",
            list(BETTING_STRATEGIES.keys()),
            index=1  # Standard by default
        )
        st.session_state.selected_strategy = strategy_name
        
        strategy = BETTING_STRATEGIES[strategy_name]
        st.info(strategy['description'])
        
        st.markdown("---")
        
        # Quick stats
        all_bets = get_all_bets()
        if not all_bets.empty:
            closed = all_bets[all_bets['status'] == 'closed']
            if not closed.empty:
                total_profit = closed['profit'].sum()
                total_staked = closed['stake'].sum()
                roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
                
                st.markdown("### 📈 Performance")
                st.metric("Paris clôturés", len(closed))
                st.metric("Profit total", f"{total_profit:+.2f}€")
                st.metric("ROI", f"{roi:.1f}%")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            <p>⚠️ Les paris sportifs comportent des risques.<br>
            Pariez de manière responsable.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tabs = st.tabs([
        "🏠 Accueil",
        "📡 Événements",
        "🎾 Prédiction",
        "💰 Bankroll",
        "🏆 Classement Elo",
        "📊 Statistiques",
        "📅 Calendrier",
        "🔄 Mise à jour"
    ])
    
    with tabs[0]:
        show_home_page()
    
    with tabs[1]:
        show_events_page()
    
    with tabs[2]:
        show_prediction_page()
    
    with tabs[3]:
        show_bankroll_page()
    
    with tabs[4]:
        show_rankings_page()
    
    with tabs[5]:
        show_stats_page()
    
    with tabs[6]:
        show_calendar_page()
    
    with tabs[7]:
        show_update_page()

if __name__ == "__main__":
    main()
