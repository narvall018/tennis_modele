"""Explicit one-click ATP quote scan, isolated paper ledger, preserved archives."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app import atp_reference_strategy as engine
from src.app import atp_reference_ledger as ledger
from src.app import tennis_strategy_ledger as old_atp
from src.app import wta_strategy_ledger as old_wta
from src.app.odds_api import active_sports, fetch_h2h_odds
from src.app.tennis_strategy_page import _bankroll_metrics, _history


def collect(root):
    """Called only on an explicit scan action, never by the refresh fragment."""
    snapshot = {'at': engine.utc().isoformat(), 'events': [], 'errors': [], 'coverage': [],
                'catalogue_count': 0, 'omitted_competitions': 0, 'remaining': None}
    try:
        catalogue = active_sports(root)
        if not catalogue.ok:
            snapshot['errors'].append(catalogue.error)
            return snapshot
        if not isinstance(catalogue.events, list):
            raise ValueError('Catalogue invalide')
        keys = sorted({s['key'] for s in catalogue.events if s.get('active') and
                       str(s.get('key', '')).startswith('tennis_atp_')})
        snapshot['catalogue_count'] = len(keys)
        snapshot['omitted_competitions'] = max(0, len(keys)-8)
        for key in keys[:8]:
            response = fetch_h2h_odds(root, key, regions='eu,fr')
            snapshot['remaining'] = response.remaining
            if not response.ok:
                snapshot['errors'].append(f'{key} : {response.error}')
                snapshot['coverage'].append({'Compétition': key, 'Réponse': 'Échec', 'Matchs': 0})
                continue
            if not isinstance(response.events, list):
                snapshot['errors'].append(f'{key} : réponse invalide.'); continue
            accepted = [e for e in response.events if isinstance(e, dict) and e.get('sport_key') == key]
            if len(accepted) != len(response.events):
                snapshot['errors'].append(f'{key} : événements sans identité de circuit cohérente exclus.')
            snapshot['coverage'].append({'Compétition': key, 'Réponse': 'OK', 'Matchs': len(accepted)})
            snapshot['events'].extend(accepted)
    except Exception as error:
        # Provider exception bodies may contain the API key: never display them.
        snapshot['errors'].append(f'Consultation interrompue ({type(error).__name__}).')
    return snapshot


@st.fragment(run_every='30s')
def _results(snapshot, db, owner):
    now = engine.utc()
    summary = ledger.state(db, owner)
    st.caption(f"Relevé du {engine.utc(snapshot['at']).tz_convert('Europe/Paris'):%d/%m/%Y %H:%M:%S} (Paris). "
               'Fraîcheur revérifiée toutes les 30 secondes et à l’enregistrement, sans nouvel appel API.')
    for error in snapshot['errors']:
        st.warning(error)
    if snapshot['omitted_competitions']:
        st.warning(f"Couverture partielle : {snapshot['omitted_competitions']} compétition(s) non interrogée(s), limite de huit par scan.")
    if snapshot['remaining'] is not None:
        st.caption(f"Quota API restant indiqué par le fournisseur : {snapshot['remaining']}.")
    if not pd.Timedelta(0) <= now-engine.utc(snapshot['at']) <= pd.Timedelta(minutes=5):
        st.warning('Relevé expiré : relancer le scan. Aucune sélection ancienne ne peut être enregistrée.')
        return
    if not snapshot['catalogue_count'] and not snapshot['errors']:
        st.info('Aucune compétition ATP active dans le catalogue du fournisseur. Aucun match n’a été évalué.')
        return
    if snapshot['coverage']:
        with st.expander('Couverture du fournisseur'):
            st.dataframe(pd.DataFrame(snapshot['coverage']), hide_index=True)
    results = [engine.analyse_event(e, now) for e in snapshot['events']]
    if not results:
        st.info('Aucun match ATP reçu à analyser. Cela ne prouve pas une absence d’opportunité sur tout le marché.')
        return
    analysed = sum(r['status'] != 'blocked' for r in results)
    blocked = len(results)-analysed
    candidates = {r['candidate']['event_key'] for r in results if r.get('candidate')}
    st.write(f'{len(results)} relevé(s) de match · {analysed} analysable(s) · {blocked} bloqué(s) / hors fenêtre · '
             f'{len(candidates)} match(s) avec signal théorique.')
    if candidates:
        st.success('Signal théorique détecté — simulation uniquement, rentabilité non démontrée.')
    elif analysed:
        st.info('Aucune opportunité selon la règle parmi les matchs analysables. Les matchs bloqués ne sont pas évalués.')
    else:
        st.warning('Analyse impossible pour les matchs reçus : consulter les motifs ci-dessous.')
    with st.expander('Tous les matchs et motifs de blocage', expanded=not bool(candidates)):
        labels = {'candidate': 'Signal théorique', 'no_signal': 'Pas de signal', 'blocked': 'Non analysable'}
        st.dataframe(pd.DataFrame([{'Match': r['match'], 'Tournoi': r['competition'],
                    'Statut': labels[r['status']], 'Bookmakers analysés': r['checked_books'],
                    'Motif': r['reason'], 'Cotes exclues': ' ; '.join(r.get('blocked_books', []))}
                    for r in results]), hide_index=True)
    planned = engine.allocation(results, summary)
    if not planned:
        if candidates:
            st.info('Les matchs sélectionnés figurent déjà dans ton carnet. Aucun doublon ajouté.')
        return
    st.dataframe(pd.DataFrame([{
        'Début Paris': engine.utc(item['candidate']['fixture']['start']).tz_convert('Europe/Paris').strftime('%d/%m %H:%M'),
        'Sélection': item['candidate']['pick'],
        'Bookmaker': engine.BOOKMAKERS[item['candidate']['fixture']['bookmaker']],
        'Cote': item['candidate']['odds'],
        'Score de référence': f"{item['candidate']['probability']:.2%}",
        'EV estimée nette': f"{item['candidate']['expected_returns'][item['candidate']['selected_side']]:+.2%}",
        'Mise simulée €': item['stake_cents']/100} for item in planned]), hide_index=True)
    st.caption('Le score de référence n’est pas une probabilité validée. Les mises proposées au total respectent '
               'le budget restant ; ordre de début des matchs. Une seule sélection par match, au meilleur avantage '
               'estimé parmi les paires françaises analysables. Une mise nulle signifie budget épuisé.')
    first = planned[0]
    if first['stake_cents'] <= 0:
        st.info('Budget quotidien ou capital disponible épuisé. Aucun nouvel enregistrement possible.')
        return
    c = first['candidate']
    st.write(f"Prochaine simulation : **{c['pick']}**, cote **{c['odds']:.2f}** chez "
             f"**{engine.BOOKMAKERS[c['fixture']['bookmaker']]}**, mise **{first['stake_cents']/100:.2f} €**.")
    if st.button('Enregistrer la sélection prioritaire en simulation', key=f'reference_record_{owner}'):
        try:
            ledger.record(db, owner, c)
            st.rerun()
        except (ValueError, KeyError, TypeError) as error:
            st.error(str(error))


def _archives(root, owner):
    st.caption('Les anciennes stratégies sont retirées de l’analyse. Leurs carnets restent séparés : '
               'consultation, règlement des simulations en attente et sauvegarde uniquement.')
    for title, filename, api, prefix in [
            ('Ancien carnet ATP — modèle statistique', 'tennis_strategy.sqlite3', old_atp, 'archive_atp'),
            ('Ancien carnet WTA — section retirée', 'wta_strategy.sqlite3', old_wta, 'archive_wta')]:
        path = root/'bets'/filename
        summary = None
        if path.exists():
            try:
                summary = api.state(path, owner)
            except ValueError:
                pass
        with st.expander(title):
            if summary is not None:
                _bankroll_metrics(summary, ledger_api=api, show_budget=False)
                _history(path, owner, summary, ledger_api=api, prefix=prefix)
            else:
                st.info('Aucun carnet de cette ancienne stratégie trouvé pour ton compte sur ce serveur.')
                backup = st.file_uploader('Restaurer une ancienne sauvegarde dans les archives',
                                          type=['json'], key=f'{prefix}_restore_{owner}')
                if backup is not None and st.button('Restaurer ce carnet archivé', key=f'{prefix}_restore_button_{owner}'):
                    try:
                        api.restore_backup(path, owner, backup.getvalue().decode('utf-8'))
                        st.rerun()
                    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
                        st.error(f'Sauvegarde refusée : {error}')
    st.caption('Aucun ancien fichier n’est supprimé. Sur Streamlit Cloud, le stockage local peut disparaître '
               'au redéploiement : les sauvegardes JSON restent importantes.')


def render_atp_reference_page(root: Path, user_id: int, username: str):
    owner = f'{int(user_id)}:{username}'
    db = root/'bets/atp_reference_price.sqlite3'
    st.title('Stratégie ATP — comparaison de cotes')
    st.warning('Rentabilité non démontrée : stratégie expérimentale, simulation uniquement. Aucun pari envoyé à un bookmaker.')
    st.caption('Référence Pinnacle corrigée de sa marge, score réduit de 0,5 point · EV estimée nette ≥ 3 % · '
               'cotes 1,30–5,00 · mise fixe 0,25 % du capital de début de journée · plafond quotidien 2 %.')
    st.info('Nouvelle bankroll, séparée de l’ancien modèle ATP et de la WTA. Aucun solde ni rendement ancien '
            'n’est transféré. Carnet privé stocké localement au serveur : sauvegarder le JSON régulièrement.')
    with st.expander('Ce que le diagnostic historique permet — et ne permet pas — de conclure'):
        st.write('Diagnostic ATP 2023–2025 : +24,38 % sur 60 paris réglés ; intervalle 95 % '
                 '[−7,95 % ; +58,79 %]. Filtre de validation non franchi. Historique déjà exploré.')
        st.write('Exécution historique aux prix Bet365, pas aux prix français actuels ; simultanéité des cotes '
                 'historiques non prouvée. En direct, Bet365 n’est pas nécessaire. Pinnacle est une référence '
                 'imparfaite et peut être retardée par le fournisseur, même avec un horodatage récent.')
    analyse, journal, archives = st.tabs(['Opportunités ATP', 'Carnet et sauvegarde', 'Archives des anciennes stratégies'])
    with archives:
        _archives(root, owner)
    try:
        summary = ledger.state(db, owner)
    except ValueError:
        with analyse:
            st.subheader('Créer la nouvelle bankroll de simulation')
            with st.form(f'reference_initial_{owner}'):
                initial = st.number_input('Capital fictif initial (€)', 10., 1_000_000., 1000., 50.)
                if st.form_submit_button('Initialiser la nouvelle simulation'):
                    try:
                        ledger.initialise(db, owner, initial)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
            backup = st.file_uploader('Restaurer une sauvegarde de la stratégie de comparaison de cotes',
                                      type=['json'], key=f'reference_restore_{owner}')
            if backup is not None and st.button('Restaurer dans ce compte vide', key=f'reference_restore_button_{owner}'):
                try:
                    ledger.restore_backup(db, owner, backup.getvalue().decode('utf-8'))
                    st.rerun()
                except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
                    st.error(f'Sauvegarde refusée : {error}')
        with journal:
            st.info('Initialiser ou restaurer la nouvelle bankroll ; les anciens carnets sont dans les archives.')
        return
    with journal:
        _history(db, owner, summary, ledger_api=ledger, prefix='reference', probability_label='Score de référence (non validé)')
    with analyse:
        _bankroll_metrics(summary, ledger_api=ledger)
        st.write('Un clic récupère les cotes et analyse automatiquement les matchs ATP disponibles. '
                 'Aucun classement, aucune surface et aucune cote Bet365 à saisir.')
        st.caption('Scan explicite pour maîtriser le quota API (régions France et Europe, jusqu’à huit compétitions). '
                   'Pas de consultation payante en boucle, ni de surveillance lorsque l’application est fermée.')
        key = f'reference_snapshot_{owner}'
        if st.button('Scanner les opportunités ATP', key=f'reference_scan_{owner}', type='primary'):
            st.session_state.pop(key, None)
            with st.spinner('Récupération des paires de cotes françaises et Pinnacle…'):
                st.session_state[key] = collect(root)
        snapshot = st.session_state.get(key)
        if snapshot:
            _results(snapshot, db, owner)
        else:
            st.info('Aucun scan effectué dans cette session. Cliquer sur « Scanner les opportunités ATP ».')
