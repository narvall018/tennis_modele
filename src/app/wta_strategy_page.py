"""WTA paper strategy UI. Reference-market inputs are never silently substituted."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.app import wta_strategy as engine
from src.app import wta_strategy_ledger as ledger
from src.app.maintenance import run_task
from src.app.odds_api import active_sports, fetch_h2h_odds
from src.app.tennis_strategy_page import _history


@st.cache_resource(show_spinner=False, max_entries=2)
def _bundle(root, fingerprint):
    return engine.load_bundle(Path(root))


@st.cache_resource(show_spinner=False, max_entries=3)
def _states(root, fingerprint, day):
    return engine.replay(_bundle(root, fingerprint)[1], pd.Timestamp(day))


@st.cache_data(show_spinner=False, ttl=900, max_entries=16)
def _score(root, fingerprint, fixture_json):
    fixture = json.loads(fixture_json)
    day = str(engine.utc(fixture['start']).tz_convert('Europe/Paris').date())
    return engine.score_fixture(_bundle(root, fingerprint), fixture,
                                state=_states(root, fingerprint, day))


def render_wta_strategy_page(root: Path, user_id: int, username: str):
    owner = f'{int(user_id)}:{username}'
    db = root / 'bets/wta_strategy.sqlite3'
    st.title('Stratégie WTA — suivi expérimental')
    st.warning('Rentabilité non démontrée. Simulation uniquement : aucun pari réel envoyé. '
               'Le filtre de validation historique n’a pas été franchi.')
    st.caption('Simple WTA tableau principal · arbres récents + surface · EV ≥ 2 % après décote de 2 % · '
               'cotes 1,30–5,00 · mise 0,25 %, plafond quotidien 2 %.')
    st.info('Bankroll WTA indépendante de l’ATP : les plafonds ne sont pas communs. '
            'Sur Streamlit Cloud, le stockage local peut disparaître au redéploiement : sauvegarder régulièrement le JSON.')
    try:
        summary = ledger.state(db, owner)
    except ValueError:
        with st.form(f'wta_initial_{owner}'):
            initial = st.number_input('Capital fictif initial WTA (€)', 10., 1_000_000., 1000., 50.)
            if st.form_submit_button('Initialiser la simulation WTA'):
                ledger.initialise(db, owner, initial)
                st.rerun()
        backup = st.file_uploader('Restaurer une sauvegarde WTA JSON', type=['json'], key=f'wta_restore_{owner}')
        if backup is not None and st.button('Restaurer WTA dans ce compte vide'):
            try:
                ledger.restore_backup(db, owner, backup.getvalue().decode('utf-8'))
                st.rerun()
            except (ValueError, KeyError, TypeError) as error:
                st.error(f'Sauvegarde WTA refusée : {error}')
        return
    values = [f"{summary['balance_cents']/100:.2f} €", f"{summary['available_cents']/100:.2f} €",
              f"{summary['reserved_cents']/100:.2f} €", f"{summary['profit_cents']/100:+.2f} €",
              '—' if summary['roi'] is None else f"{summary['roi']:+.2%}"]
    for col, label, value in zip(st.columns(5), ['Bankroll WTA simulée', 'Disponible', 'Mises ouvertes', 'Résultat net', 'ROI sur mises'], values):
        col.metric(label, value)
    st.caption(f"Budget WTA restant aujourd’hui : {summary['day_remaining_cents']/100:.2f} € ; "
               f"prochaine mise maximale : {ledger.proposed_stake(summary)/100:.2f} €. Journée Europe/Paris.")
    analyse, journal = st.tabs(['Analyser un match WTA', 'Carnet et sauvegarde WTA'])
    with journal:
        _history(db, owner, summary, ledger_api=ledger, prefix='wta')
    with analyse:
        try:
            fingerprint = engine.digest(root / 'models/wta_live_strategy/metadata.json')
            bundle = _bundle(str(root), fingerprint)
            meta, history, _, _ = bundle
        except (ValueError, KeyError, OSError) as error:
            st.error(f'Paquet WTA indisponible ({type(error).__name__}). Préparer le modèle WTA avant de calculer.')
            st.code('python3 scripts/prepare_wta_strategy.py', language='bash')
            return
        evidence = meta['evidence']
        lower, upper = evidence['uncertainty']['ci95']
        st.caption(f"Diagnostic 2023–2025 : {evidence['roi']['0.02']:+.2%} sur {evidence['settled']} paris réglés ; "
                   f"intervalle 95 % [{lower:+.2%} ; {upper:+.2%}]. Historique Bet365/Pinnacle, pas un rendement attendu en France.")
        st.markdown(f"Historique jusqu’au **{meta['history_last_date']}** ; modèle annuel **{meta['model_year']}**.")
        reasons = engine.freshness_reasons(meta)
        for reason in reasons:
            st.error(reason)
        if reasons:
            if st.button('Actualiser les données de la stratégie WTA', key=f'wta_refresh_{owner}'):
                with st.spinner('Téléchargement WTA et contrôles, sans réentraînement…'):
                    result = run_task(root, 'wta_strategy_refresh')
                if result['ok']:
                    _bundle.clear(); _states.clear(); _score.clear()
                    st.rerun()
                st.error('Actualisation refusée ; le carnet reste accessible.')
                st.code(result['output'], language='text')
            return
        st.info('Le modèle exige deux cotes Bet365 ET deux cotes Pinnacle du même match. '
                'Elles servent uniquement de références, pas de lieux où parier. Les deux cotes françaises '
                'servent à la simulation. Six cotes réelles, relevées à moins de cinq minutes d’écart, sont nécessaires. '
                'Sans référence Bet365 disponible, aucun calcul : ne pas lui substituer un autre bookmaker.')
        if st.button('Consulter les cotes WTA françaises et Pinnacle (quota API)', key=f'wta_fetch_{owner}'):
            with st.spinner('Consultation explicite du fournisseur…'):
                catalogue = active_sports(root)
                prices, errors, events = [], [], 0
                keys = [s['key'] for s in catalogue.events if catalogue.ok and s.get('active')
                        and str(s.get('key', '')).startswith('tennis_wta_') and not s.get('has_outrights')]
                if not catalogue.ok:
                    errors.append(catalogue.error)
                for key in keys[:8]:
                    response = fetch_h2h_odds(root, key, regions='eu,fr')
                    if response.ok:
                        events += len(response.events)
                        prices.extend(engine.quotes(response.events))
                    else:
                        errors.append(response.error)
                st.session_state[f'wta_prices_{owner}'] = prices
                if errors:
                    st.warning(' ; '.join(errors))
                st.session_state[f'wta_diagnostic_{owner}'] = (
                    f'{len(keys)} compétition(s) WTA active(s) dans le catalogue ; {events} match(s) renvoyé(s) ; '
                    f'{len(prices)} paire(s) récente(s) acceptée(s). Bet365 reste à saisir manuellement.')
        if f'wta_diagnostic_{owner}' in st.session_state:
            st.caption(st.session_state[f'wta_diagnostic_{owner}'])
        prices = st.session_state.get(f'wta_prices_{owner}', [])
        french = [p for p in prices if p['bookmaker'] in engine.BOOKMAKERS]
        selected = pinnacle = None
        if prices:
            st.dataframe(pd.DataFrame(prices), hide_index=True)
        if french:
            index = st.selectbox('Relevé français WTA à analyser', [-1, *range(len(french))],
                     format_func=lambda i: 'Saisie manuelle' if i < 0 else
                     f"{french[i]['player_1']} — {french[i]['player_2']} / {engine.BOOKMAKERS[french[i]['bookmaker']]}")
            if index >= 0:
                selected = french[index]
                pinnacle = next((p for p in prices if p['event_id'] == selected['event_id'] and p['bookmaker'] == 'pinnacle'
                                 and p['player_1'] == selected['player_1'] and p['player_2'] == selected['player_2']), None)
        today = engine.utc().tz_convert('Europe/Paris').tz_localize(None).normalize()
        past = history[history['_date'] < today]
        players = sorted(set(past['_p1']) | set(past['_p2']))
        local = datetime.now(ZoneInfo('Europe/Paris'))
        start = engine.utc(selected['start']).tz_convert('Europe/Paris').to_pydatetime() if selected else local + timedelta(hours=2)
        context = selected['event_id'] + selected['bookmaker'] if selected else 'manual'
        with st.form(f'wta_match_{owner}_{context}'):
            left, right = st.columns(2)
            fields = {}
            for side, col in [(1, left), (2, right)]:
                with col:
                    fields[f'player_{side}'] = st.selectbox(f'Joueuse côté {side}', players, index=None)
                    fields[f'player_{side}_rank'] = st.number_input(f'Classement WTA actuel — côté {side}', 1, 3000, value=None)
                    fields[f'odds_{side}'] = st.number_input(f'Cote française — côté {side}', 1.01, 100.,
                        value=float(selected[f'odds_{side}']) if selected else None, step=.01, disabled=selected is not None)
                    fields[f'bet365_odds_{side}'] = st.number_input(f'Référence Bet365 — côté {side}', 1.01, 100., value=None, step=.01)
                    fields[f'pinnacle_odds_{side}'] = st.number_input(f'Référence Pinnacle — côté {side}', 1.01, 100.,
                        value=float(pinnacle[f'odds_{side}']) if pinnacle else None, step=.01, disabled=pinnacle is not None)
            surface = st.selectbox('Surface WTA confirmée', ['Hard', 'Clay', 'Grass'], index=None)
            tournament = st.text_input('Tournoi WTA confirmé', value=selected['competition'] if selected else '')
            book = st.selectbox('Bookmaker français WTA', list(engine.BOOKMAKERS),
                index=list(engine.BOOKMAKERS).index(selected['bookmaker']) if selected else 0,
                format_func=engine.BOOKMAKERS.get, disabled=selected is not None)
            day = st.date_input('Date de début WTA (Paris)', start.date(), disabled=selected is not None)
            clock = st.time_input('Heure de début WTA (Paris)', start.time().replace(tzinfo=None), disabled=selected is not None)
            confirmed = st.checkbox('Je confirme le simple WTA tableau principal, les deux identités dans cet ordre, '
                'la surface et les classements. Toutes les cotes saisies manuellement viennent d’être relevées sur ce même match ; '
                'Bet365 et Pinnacle ne sont pas des prix substitués.')
            submitted = st.form_submit_button('Calculer selon la stratégie WTA')
        if submitted:
            st.session_state.pop(f'wta_candidate_{owner}', None)
            try:
                if not confirmed or any(value is None for value in fields.values()):
                    raise ValueError('Renseigner et confirmer les deux joueuses, classements et six cotes réelles.')
                stamp = engine.utc().isoformat()
                fixture = {**fields, 'tour': 'WTA', 'singles_main_draw': True,
                    'surface': surface, 'tournament': tournament, 'bookmaker': book,
                    'start': selected['start'] if selected else datetime.combine(day, clock, tzinfo=ZoneInfo('Europe/Paris')).isoformat(),
                    'quote_at': selected['quote_at'] if selected else stamp,
                    'bet365_quote_at': stamp, 'pinnacle_quote_at': pinnacle['quote_at'] if pinnacle else stamp,
                    'api_pair': selected, 'pinnacle_api_pair': pinnacle,
                    'price_source': 'manual_confirmed_references_with_optional_api_pairs'}
                with st.spinner('Reconstruction des statistiques antérieures et calcul WTA…'):
                    candidate = _score(str(root), fingerprint, json.dumps(fixture, sort_keys=True))
                st.session_state[f'wta_candidate_{owner}'] = candidate
            except (ValueError, TypeError, KeyError, OSError) as error:
                st.error(str(error))
        candidate = st.session_state.get(f'wta_candidate_{owner}')
        if candidate:
            st.markdown(f"Dernier calcul soumis : **{candidate['fixture']['player_1']} — {candidate['fixture']['player_2']}**.")
            st.caption('Les modifications non recalculées du formulaire ne changent pas ce résultat.')
            st.dataframe(pd.DataFrame({'Joueuse': [candidate['fixture']['player_1'], candidate['fixture']['player_2']],
                'Probabilité modèle': candidate['probabilities'], 'EV estimée aux cotes françaises': candidate['expected_returns']}), hide_index=True)
            st.info(candidate['reason'])
            if candidate['eligible']:
                st.markdown(f"Sélection théorique : **{candidate['pick']}**, cote **{candidate['odds']:.2f}**.")
                if st.button('Enregistrer ce pari WTA en simulation'):
                    try:
                        if candidate['model_sha256'] != meta['files']['booster.ubj'] or candidate['history_sha256'] != meta['files']['history.csv.gz']:
                            raise ValueError('Paquet modifié : recalculer la sélection WTA.')
                        ledger.record(db, owner, candidate)
                        st.session_state.pop(f'wta_candidate_{owner}', None)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
