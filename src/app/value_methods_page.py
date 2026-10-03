"""Separate football/UFC pages: automatic inputs, calculations and private journals."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app import value_methods as engine
from src.app import value_methods_data as data
from src.app import value_methods_ledger as ledger


def _journal(db, owner, sport):
    try:
        summary = ledger.state(db, owner)
    except ValueError:
        st.info('Initialiser ou restaurer une bankroll de simulation pour utiliser le carnet.')
        return
    bets = pd.DataFrame(summary['bets'])
    if not bets.empty:
        display = bets[['id', 'created_at', 'start_at', 'pick', 'odds', 'probability', 'stake_cents', 'status', 'profit_cents']].copy()
        display[['stake_cents', 'profit_cents']] /= 100
        st.dataframe(display.rename(columns={'id': 'N°', 'created_at': 'Ajout UTC', 'start_at': 'Début UTC',
            'pick': 'Sélection', 'odds': 'Cote', 'probability': 'Score de référence' if sport == 'football' else 'Probabilité modèle prudente',
            'stake_cents': 'Mise €', 'status': 'État', 'profit_cents': 'Résultat €'}), hide_index=True)
        pending = [b for b in summary['bets'] if b['status'] == 'pending']
        if pending:
            with st.form(f'value_settle_{sport}_{owner}'):
                chosen = st.selectbox('Simulation à régler', pending, format_func=lambda b: f"#{b['id']} — {b['pick']}")
                result = st.selectbox('Résultat confirmé', ['won', 'lost', 'void'],
                                      format_func=lambda r: {'won': 'Gagné', 'lost': 'Perdu', 'void': 'Annulé'}[r])
                if st.form_submit_button('Enregistrer le résultat'):
                    try:
                        ledger.settle(db, owner, chosen['id'], result)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
        st.download_button('Exporter le carnet CSV', display.to_csv(index=False).encode('utf-8-sig'),
                           f'{sport}_simulations.csv', 'text/csv', key=f'value_csv_{sport}_{owner}')
    else:
        st.info('Aucune simulation enregistrée.')
    st.caption('Résultats confirmés puis saisis dans ce carnet ; gains simulés après décote de 2 %. '
               'Un nul football est gagnant si la sélection était « Match nul ». Carnet privé au compte, stocké sur le serveur.')
    st.download_button('Sauvegarder la bankroll et le carnet JSON', ledger.export_backup(db, owner, sport),
                       f'{sport}_simulation_sauvegarde.json', 'application/json', key=f'value_backup_{sport}_{owner}')


def _account(db, owner, sport):
    try:
        return ledger.state(db, owner)
    except ValueError:
        with st.expander('Créer ou restaurer le carnet de simulation'):
            with st.form(f'value_initial_{sport}_{owner}'):
                amount = st.number_input('Capital fictif initial (€)', 10., 1_000_000., 1000., 50., key=f'value_capital_{sport}_{owner}')
                if st.form_submit_button('Initialiser la simulation'):
                    try:
                        ledger.initialise(db, owner, amount)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
            backup = st.file_uploader('Restaurer une sauvegarde JSON dans ce compte vide', type=['json'], key=f'value_restore_{sport}_{owner}')
            if backup is not None and st.button('Restaurer le carnet', key=f'value_restore_button_{sport}_{owner}'):
                try:
                    ledger.restore_backup(db, owner, backup.getvalue().decode('utf-8'), sport)
                    st.rerun()
                except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
                    st.error(f'Sauvegarde refusée : {error}')
        return None


def _metrics(summary):
    values = [f"{summary['balance_cents']/100:.2f} €", f"{summary['available_cents']/100:.2f} €",
              f"{summary['reserved_cents']/100:.2f} €", f"{summary['profit_cents']/100:+.2f} €",
              '—' if summary['roi'] is None else f"{summary['roi']:+.2%}"]
    for col, label, value in zip(st.columns(5), ['Bankroll simulée', 'Disponible', 'Mises ouvertes', 'Résultat net', 'ROI sur mises'], values):
        col.metric(label, value)


@st.fragment(run_every='60s')
def _opportunities(root_text, sport, owner, leagues, automatic):
    root = Path(root_text)
    now = engine.utc()
    bundle = cards = None
    if sport == 'ufc':
        if automatic and data.refresh_due(root, now):
            # A failed source is retried after one hour, never every fragment tick.
            retry_path = root / 'bets/value_methods_runtime/ufc_refresh_attempt.json'
            attempt = data.read_json(retry_path, {})
            last = attempt.get('at')
            if not last or now - engine.utc(last) >= pd.Timedelta(hours=1):
                data.atomic_json(retry_path, {'at': now.isoformat()})
                try:
                    with st.spinner('Actualisation du programme officiel et des statistiques UFC…'):
                        data.refresh_ufc(root)
                except Exception as error:
                    data.atomic_json(retry_path, {'at': now.isoformat(), 'error': type(error).__name__})
                    st.warning(f'Actualisation UFC interrompue ({type(error).__name__}). Les données périmées restent bloquées.')
        try:
            bundle, cards = data.load_ufc(root)
            st.caption(f"Historique UFC jusqu’au {bundle['history_last_date']} · {bundle['history_fights_used']} combats · "
                       f"dernière vérification : {engine.utc(bundle['checked_at']).tz_convert('Europe/Paris'):%d/%m/%Y %H:%M} (Paris).")
        except (ValueError, KeyError, TypeError) as error:
            st.error(str(error))
            return
    snapshot = data.collect_live(root, sport, leagues, now) if automatic else data.read_quote_cache(root, sport)
    # Quotes can update while the network request is running. Assess them at
    # receipt, not against the timestamp from before the collection started.
    now = engine.utc()
    if not snapshot or snapshot.get('sports') != (list(leagues) if sport == 'football' else [engine.MMA_KEY]):
        st.info('Aucun relevé pour ces compétitions. Lancer le scan ou activer l’actualisation automatique.')
        return
    st.caption(f"Dernier scan : {engine.utc(snapshot['at']).tz_convert('Europe/Paris'):%d/%m/%Y %H:%M:%S} (Paris).")
    for error in snapshot.get('errors', []):
        st.warning(error)
    if not pd.Timedelta(0) <= now - engine.utc(snapshot['at']) <= pd.Timedelta(minutes=5):
        st.info('Le relevé a plus de cinq minutes : aucune ancienne cote n’est proposée. '
                'Un scan manuel permet de chercher des prix récents ; le scan automatique reprend après une heure.')
        return
    results = [engine.analyse_event(sport, event, now, bundle, cards) for event in snapshot['events']]
    if not results:
        st.info('Aucun événement reçu à analyser.')
        return
    analysed = sum(r['status'] != 'blocked' for r in results)
    candidates = sum(r['candidate'] is not None for r in results)
    st.write(f'{len(results)} événement(s) reçu(s) · {analysed} analysable(s) · {candidates} signal(aux) théorique(s).')
    if candidates:
        st.success('Signal théorique détecté — rentabilité actuelle non démontrée.')
    elif analysed:
        st.info('Aucune cote analysable ne satisfait la méthode actuellement.')
    else:
        st.warning('Analyse bloquée pour les événements reçus : consulter les motifs.')
    with st.expander('Événements analysés et motifs', expanded=not bool(candidates)):
        st.dataframe(pd.DataFrame([{'Rencontre': r['match'], 'Compétition': r['competition'],
            'État': {'candidate': 'Signal théorique', 'blocked': 'Bloqué', 'no_signal': 'Pas de signal'}[r['status']],
            'Marchés analysés': r['checked_books'], 'Motif': r['reason'],
            'Cotes exclues': ' ; '.join(r.get('blocked_books', []))} for r in results]), hide_index=True)
    db = ledger.database(root, sport)
    try:
        summary = ledger.state(db, owner)
    except ValueError:
        summary = None
    if not summary:
        if candidates:
            st.dataframe(pd.DataFrame([{'Sélection': r['candidate']['pick'], 'Cote': r['candidate']['odds'],
                'EV estimée nette': f"{r['candidate']['expected_returns'][r['candidate']['selected_side']]:+.2%}"}
                for r in results if r['candidate']]), hide_index=True)
            st.info('Créer le carnet ci-dessus pour enregistrer des simulations.')
        return
    planned = engine.allocation(results, summary)
    if not planned:
        if candidates:
            st.info('Les sélections détectées figurent déjà dans ton carnet.')
        return
    st.dataframe(pd.DataFrame([{'Début Paris': engine.utc(c['candidate']['fixture']['start']).tz_convert('Europe/Paris').strftime('%d/%m %H:%M'),
        'Sélection': c['candidate']['pick'], 'Source de cote': engine.BOOKMAKERS[c['candidate']['fixture']['bookmaker']],
        'Cote': c['candidate']['odds'], 'Score / probabilité prudente': f"{c['candidate']['probability']:.2%}",
        'EV estimée nette': f"{c['candidate']['expected_returns'][c['candidate']['selected_side']]:+.2%}",
        'Mise simulée €': c['stake_cents']/100} for c in planned]), hide_index=True)
    if sport == 'ufc':
        with st.expander('Probabilités UFC selon la durée et la méthode de victoire'):
            details = []
            for item in planned:
                candidate = item['candidate']
                for key in ('three_rounds', 'five_rounds'):
                    model = candidate['model'][key]
                    for side in (0, 1):
                        p = model['method_probabilities'][side*3:side*3+3]
                        details.append({'Combattant': candidate['fixture']['outcomes'][side], 'Rounds': model['rounds'],
                                        'Victoire': model['win_probabilities'][side], 'KO/TKO': p[0], 'Soumission': p[1], 'Décision': p[2]})
            st.dataframe(pd.DataFrame(details), hide_index=True)
    first = planned[0]
    if first['stake_cents'] > 0 and st.button('Enregistrer la sélection prioritaire en simulation', key=f'value_record_{sport}_{owner}'):
        try:
            ledger.record(root, sport, owner, first['candidate'])
            st.rerun()
        except (ValueError, KeyError, TypeError) as error:
            st.error(str(error))
    elif first['stake_cents'] <= 0:
        st.info('Budget quotidien ou capital disponible épuisé.')


def render_value_method_page(root: Path, sport: str, user_id: int, username: str):
    owner = f'{int(user_id)}:{username}'
    label = 'Football' if sport == 'football' else 'UFC'
    st.title(f'Stratégie {label} — ' + ('écarts de cotes' if sport == 'football' else 'modèle Markov'))
    st.caption('Méthode expérimentale · actualisation et calculs automatiques · carnet de simulation séparé par sport et par compte.')
    st.info('Les signaux sont théoriques : la rentabilité actuelle n’est pas prouvée. '
            'Les mises du carnet sont fictives et aucun pari n’est envoyé à un opérateur.')
    with st.expander('Méthode et résultats de recherche'):
        if sport == 'football':
            st.write('Comparer les cotes françaises au marché Pinnacle 1X2 corrigé de sa marge. '
                     'Retenir le plus faible des scores proportionnel et puissance, retrancher 0,5 point, '
                     'puis exiger une EV estimée nette d’au moins 3 % après décote de 2 %. Le score n’est pas une probabilité validée.')
            st.write('Diagnostic exploratoire 2021–2024 : +14,50 % sur 499 paris aux meilleurs prix historiques anonymes ; '
                     'les prix réellement accessibles ne sont pas certifiés. La validation statistique corrigée n’est pas franchie. '
                     'Ces résultats ne sont pas ceux des prix français en direct.')
        else:
            st.write('Le modèle fait évoluer le combat entre position debout, contrôle au sol et victoires par KO ou soumission. '
                     'Les taux viennent des statistiques UFC antérieures ; le jugement des décisions est ajusté sur des caractéristiques calculées avant chaque carte. '
                     'Le favori doit rester le même et dépasser 26 % d’EV estimée nette en trois comme en cinq rounds.')
            st.write('Cette version est une adaptation expérimentale. Le modèle bayésien de Holmes, McHale et Zychaluk '
                     'a rapporté +15,27 % sur 144 paris en 2018 ; sa rentabilité n’a pas été reproduite par cette application.')
            st.markdown('[Étude originale](https://doi.org/10.1016/j.ijforecast.2022.01.007)')
    st.caption('Cotes 1,30–5,00 · mises simulées 0,25 % du capital au début de la journée · plafond quotidien 2 % · '
               'prix de moins de cinq minutes · une sélection par rencontre.')
    db = ledger.database(root, sport)
    summary = _account(db, owner, sport)
    if summary:
        _metrics(summary)
    opportunities, journal = st.tabs(['Opportunités et automatisation', 'Carnet et sauvegarde'])
    with journal:
        _journal(db, owner, sport)
    with opportunities:
        leagues = []
        if sport == 'football':
            leagues = st.multiselect('Compétitions à scanner', list(engine.FOOTBALL_SPORTS), default=data.DEFAULT_LEAGUES,
                format_func=engine.FOOTBALL_SPORTS.get, key=f'value_leagues_{sport}_{owner}')
            if not leagues:
                st.info('Sélectionner au moins une compétition.')
                return
        automatic = st.toggle('Actualisation automatique', value=True, key=f'value_auto_{sport}_{owner}')
        st.caption('À l’ouverture puis une fois par heure tant que cette page est active. '
                   'Toutes les 60 secondes, les prix sont revérifiés. Le carnet reste sous ton contrôle.')
        if sport == 'ufc' and st.button('Actualiser les statistiques et le programme UFC', key=f'value_refresh_{owner}'):
            try:
                with st.spinner('Vérification UFCStats et reconstruction du modèle…'):
                    data.refresh_ufc(root)
                st.success('Données UFC actualisées.')
            except Exception as error:
                st.error(f'Actualisation interrompue ({type(error).__name__}).')
        if st.button(f'Scanner maintenant — {label}', type='primary', key=f'value_scan_{sport}_{owner}'):
            with st.spinner('Récupération des cotes et contrôles…'):
                data.collect_live(root, sport, leagues if sport == 'football' else None, force=True)
        _opportunities(str(root), sport, owner, leagues, automatic)


def render_football_value_page(root: Path, user_id: int, username: str):
    render_value_method_page(root, 'football', user_id, username)


def render_ufc_markov_page(root: Path, user_id: int, username: str):
    render_value_method_page(root, 'ufc', user_id, username)
