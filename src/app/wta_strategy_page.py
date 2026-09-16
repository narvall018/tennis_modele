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
from src.app.tennis_strategy_page import _bankroll_metrics, _history


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
    st.warning('Rentabilité non démontrée. Cette section propose uniquement des simulations selon '
               'la dernière règle étudiée ; aucun pari n’est envoyé à un bookmaker.')
    st.caption('WTA féminin, simple, tableau principal · arbres pondérés vers les matchs récents · '
               'EV estimée ≥ 2 % après décote de 2 % · cotes 1,30–5,00 · mise fixe 0,25 % · plafond quotidien 2 %.')
    st.info('Carnet privé à ton compte, séparé de l’ATP et des anciens paris : les bankrolls et plafonds ne sont pas communs. '
            'Stockage local au serveur : sur Streamlit Cloud, un redéploiement peut le supprimer. '
            'Télécharger régulièrement la sauvegarde JSON ; elle peut être restaurée dans une bankroll WTA vide. '
            'Aucun carnet n’est publié sur GitHub.')
    try:
        summary = ledger.state(db, owner)
    except ValueError:
        st.subheader('Créer la bankroll de simulation')
        with st.form(f'wta_initial_{owner}'):
            initial = st.number_input('Capital fictif initial (€)', 10., 1_000_000., 1000., 50., key=f'wta_capital_{owner}')
            if st.form_submit_button('Initialiser la simulation'):
                try:
                    ledger.initialise(db, owner, initial)
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        backup = st.file_uploader('Ou restaurer une sauvegarde JSON de cette stratégie', type=['json'], key=f'wta_restore_{owner}')
        if backup is not None and st.button('Restaurer dans ce compte vide', key=f'wta_restore_button_{owner}'):
            try:
                ledger.restore_backup(db, owner, backup.getvalue().decode('utf-8'))
                st.rerun()
            except (ValueError, KeyError, TypeError) as error:
                st.error(f'Sauvegarde WTA refusée : {error}')
        return
    _bankroll_metrics(summary, ledger_api=ledger)
    analyse, journal = st.tabs(['Analyser un match', 'Carnet et sauvegarde'])
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
        st.caption(f"Diagnostic historique 2023–2025 : {evidence['roi']['0.02']:+.2%} sur {evidence['settled']} paris réglés ; "
                   f"intervalle 95 % [{lower:+.2%} ; {upper:+.2%}]. Filtre de validation non franchi. "
                   'Les prix français constituent une nouvelle expérience, pas une reproduction prouvée de Bet365/Pinnacle.')
        st.markdown(f"Historique disponible jusqu’au **{meta['history_last_date']}** ; modèle annuel **{meta['model_year']}**.")
        reasons = engine.freshness_reasons(meta)
        for reason in reasons:
            st.error(reason)
        if reasons:
            st.caption('Le carnet reste utilisable. La mise à jour télécharge les résultats, contrôle les données et conserve le modèle annuel ; aucune modification de la bankroll.')
            if st.button('Actualiser les données de la stratégie WTA', key=f'wta_refresh_{owner}'):
                with st.spinner('Téléchargement WTA et contrôles, sans réentraînement…'):
                    result = run_task(root, 'wta_strategy_refresh')
                if result['ok']:
                    _bundle.clear(); _states.clear(); _score.clear()
                    st.rerun()
                st.error('Actualisation refusée ; le carnet reste accessible.')
                st.code(result['output'], language='text')
            with st.expander('Commandes pour une installation locale'):
                st.code('python3 scripts/refresh_wta_strategy.py', language='bash')
            return
        st.caption('Même parcours que l’ATP : consulter ou saisir les cotes, confirmer le match, calculer, puis enregistrer en simulation. '
                   'Spécificité WTA : les références Bet365/Pinnacle restent obligatoires dans le bloc dédié du formulaire.')
        st.caption('La consultation API demande les régions France et Europe pour récupérer aussi Pinnacle, si disponible. Bet365 reste à saisir manuellement.')
        if st.button('Consulter les cotes WTA françaises (consomme du quota API)', key=f'wta_fetch_{owner}'):
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
        if f'wta_diagnostic_{owner}' in st.session_state and not french:
            st.info('Aucune paire de cotes WTA française récente obtenue. La saisie manuelle reste possible ; aucun prix n’est inventé.')
        if french:
            st.dataframe(pd.DataFrame(french), hide_index=True)
            index = st.selectbox('Relevé à analyser (identités et contexte à confirmer)', [-1, *range(len(french))],
                     key=f'wta_quote_choice_{owner}',
                     format_func=lambda i: 'Saisie manuelle' if i < 0 else
                     f"{french[i]['player_1']} — {french[i]['player_2']} / {engine.BOOKMAKERS[french[i]['bookmaker']]}")
            if index >= 0:
                selected = french[index]
                pinnacle = next((p for p in prices if p['event_id'] == selected['event_id'] and p['bookmaker'] == 'pinnacle'
                                 and p['player_1'] == selected['player_1'] and p['player_2'] == selected['player_2']), None)
        today = engine.utc().tz_convert('Europe/Paris').tz_localize(None).normalize()
        past = history[history['_date'] < today]
        players = sorted(set(past['_p1']) | set(past['_p2']))
        if len(players) < 2:
            st.error('Historique de joueuses incomplet.')
            return
        local = datetime.now(ZoneInfo('Europe/Paris'))
        start = engine.utc(selected['start']).tz_convert('Europe/Paris').to_pydatetime() if selected else local + timedelta(hours=2)
        context = selected['event_id'] + selected['bookmaker'] if selected else 'manual'
        st.caption('Confirmer les identités exactes, la surface et les classements actuels. '
                   'Le fournisseur de cotes ne fournit pas tous ces champs. Aucun rapprochement flou automatique.')
        with st.form(f'wta_match_{owner}_{context}'):
            left, right = st.columns(2)
            fields = {}
            for side, col in [(1, left), (2, right)]:
                with col:
                    fields[f'player_{side}'] = st.selectbox(f'Joueuse correspondant au côté {side}', players, index=None)
                    fields[f'player_{side}_rank'] = st.number_input(f'Classement WTA actuel — côté {side}', 1, 3000, value=None)
                    fields[f'odds_{side}'] = st.number_input(f'Cote côté {side}', 1.01, 100.,
                        value=float(selected[f'odds_{side}']) if selected else None, step=.01, disabled=selected is not None)
            tournaments = sorted(past['_tournament'].dropna().unique().tolist())
            tournament = st.selectbox('Tournoi (libellé historique ou nouveau)', tournaments,
                index=None, accept_new_options=True, placeholder=selected['competition'] if selected else 'Choisir ou saisir le tournoi')
            surface = st.selectbox('Surface confirmée', ['Hard', 'Clay', 'Grass'], index=None)
            book = st.selectbox('Bookmaker français', list(engine.BOOKMAKERS),
                index=list(engine.BOOKMAKERS).index(selected['bookmaker']) if selected else 0,
                format_func=engine.BOOKMAKERS.get, disabled=selected is not None)
            day = st.date_input('Date de début (Paris)', start.date(), disabled=selected is not None)
            clock = st.time_input('Heure de début (Paris)', start.time().replace(tzinfo=None), disabled=selected is not None)
            with st.expander('Références du modèle WTA — Bet365 et Pinnacle (obligatoires)', expanded=True):
                st.info('Le modèle exige deux cotes Bet365 ET deux cotes Pinnacle du même match, en plus des deux cotes françaises. '
                        'Ces références ne sont pas des lieux où parier. Les six cotes réelles doivent être relevées à moins de cinq minutes d’écart. '
                        'Sans référence disponible, aucun calcul : ne pas lui substituer un autre bookmaker.')
                for side, col in enumerate(st.columns(2), 1):
                    with col:
                        fields[f'bet365_odds_{side}'] = st.number_input(f'Référence Bet365 — côté {side}', 1.01, 100., value=None, step=.01)
                        fields[f'pinnacle_odds_{side}'] = st.number_input(f'Référence Pinnacle — côté {side}', 1.01, 100.,
                            value=float(pinnacle[f'odds_{side}']) if pinnacle else None, step=.01, disabled=pinnacle is not None)
            confirmed = st.checkbox('Je confirme le simple WTA tableau principal, les deux identités dans cet ordre, '
                'la surface et les classements. Toutes les cotes saisies manuellement viennent d’être relevées sur ce même match ; '
                'Bet365 et Pinnacle ne sont pas des prix substitués.')
            submitted = st.form_submit_button('Calculer selon la stratégie figée')
        if submitted:
            st.session_state.pop(f'wta_candidate_{owner}', None)
            try:
                if not confirmed or not tournament or not surface or any(value is None for value in fields.values()):
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
            st.write(f"Calcul enregistré : **{candidate['fixture']['player_1']} — {candidate['fixture']['player_2']}**, "
                     f"{candidate['fixture']['tournament']} / {candidate['fixture']['start']}.")
            st.caption('Ce résultat correspond aux champs soumis ci-dessus, pas à des modifications non recalculées.')
            st.dataframe(pd.DataFrame({'Joueuse': [candidate['fixture']['player_1'], candidate['fixture']['player_2']],
                'Probabilité modèle': candidate['probabilities'], 'EV estimée nette': candidate['expected_returns']}), hide_index=True)
            st.info(candidate['reason'])
            if candidate['eligible']:
                st.markdown(f"Sélection théorique : **{candidate['pick']}**, cote **{candidate['odds']:.2f}**.")
                st.caption(f"Mise théorique au maximum : {ledger.proposed_stake(summary)/100:.2f} € ; montant revérifié à l’enregistrement.")
                if st.button('Enregistrer ce pari en simulation', key=f'wta_record_{owner}'):
                    try:
                        if candidate['model_sha256'] != meta['files']['booster.ubj'] or candidate['history_sha256'] != meta['files']['history.csv.gz']:
                            raise ValueError('Paquet modifié : recalculer la sélection WTA.')
                        amount = ledger.record(db, owner, candidate)
                        st.session_state.pop(f'wta_candidate_{owner}', None)
                        st.success(f'Simulation enregistrée : {amount:.2f} €. Consulter le carnet.')
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
