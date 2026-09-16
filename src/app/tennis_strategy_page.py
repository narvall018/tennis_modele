"""ATP prospective paper-trading page, separate from the app's legacy Elo picks."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.app import tennis_strategy as engine
from src.app import tennis_strategy_ledger as ledger
from src.app.odds_api import active_sports, fetch_h2h_odds
from src.app.maintenance import run_task


@st.cache_resource(show_spinner=False)
def _bundle(root_text, metadata_hash):
    return engine.load_bundle(Path(root_text))


@st.cache_data(show_spinner=False, ttl=900, max_entries=16)
def _score(root_text, metadata_hash, fixture_json):
    return engine.score_fixture(_bundle(root_text, metadata_hash), json.loads(fixture_json))


def _history(db, owner, summary):
    bets = pd.DataFrame(summary['bets'])
    st.subheader('Carnet de simulation')
    st.caption('Résultats saisis manuellement, gains simulés après décote de 2 %. '
               'Abandon ou walkover : annuler, conformément au scénario du test.')
    if not bets.empty:
        table = bets[['id', 'created_at', 'start_at', 'pick', 'odds', 'probability',
                      'stake_cents', 'status', 'profit_cents']].copy()
        table['stake_cents'] /= 100
        table['profit_cents'] /= 100
        st.dataframe(table.rename(columns={'id': 'N°', 'created_at': 'Enregistré UTC',
                     'start_at': 'Début UTC', 'pick': 'Sélection', 'odds': 'Cote',
                     'probability': 'Probabilité modèle', 'stake_cents': 'Mise €',
                     'status': 'Statut', 'profit_cents': 'Résultat €'}), hide_index=True)
        settled = bets[bets['settled_at'].notna()].sort_values(['settled_at', 'id'])
        if not settled.empty:
            curve = pd.DataFrame({'Date': pd.to_datetime(settled['settled_at']),
                                  'Bankroll simulée €': (summary['initial_cents'] + settled['profit_cents'].cumsum()) / 100})
            st.line_chart(curve.set_index('Date'))
        pending = bets[bets['status'] == 'pending']
        if not pending.empty:
            with st.form(f'atp_settle_{owner}'):
                by_id = pending.set_index('id')
                bet_id = st.selectbox('Pari à régler', list(by_id.index),
                                      format_func=lambda i: f"#{i} — {by_id.loc[i, 'pick']}")
                result = st.selectbox('Résultat confirmé', ['won', 'lost', 'void'],
                                      format_func=lambda v: {'won':'Gagné', 'lost':'Perdu', 'void':'Annulé / abandon'}[v])
                confirmed = st.checkbox('J’ai vérifié le résultat final ; cette saisie sera définitive.')
                if st.form_submit_button('Enregistrer le résultat'):
                    try:
                        if not confirmed:
                            raise ValueError('Confirmer le résultat final avant de l’enregistrer.')
                        ledger.settle(db, owner, int(bet_id), result)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
        st.download_button('Exporter le carnet CSV', table.to_csv(index=False).encode('utf-8-sig'),
                           'atp_simulation.csv', 'text/csv', key=f'atp_csv_{owner}')
    else:
        st.info('Aucune simulation enregistrée. Une absence de sélection est un résultat normal.')
    st.download_button('Sauvegarder bankroll + carnet (JSON)', ledger.export_backup(db, owner),
                       'atp_bankroll_sauvegarde.json', 'application/json', key=f'atp_backup_{owner}')


def render_tennis_strategy_page(root: Path, user_id: int, username: str):
    owner = f'{int(user_id)}:{username}'
    db = root / 'bets/tennis_strategy.sqlite3'
    st.title('Stratégie ATP — suivi expérimental')
    st.warning('Rentabilité non démontrée. Cette section propose uniquement des simulations selon '
               'la dernière règle étudiée ; aucun pari n’est envoyé à un bookmaker.')
    st.caption('ATP masculin, simple, tableau principal · arbres pondérés vers les matchs récents · '
               'EV estimée ≥ 2 % après décote de 2 % · cotes 1,30–5,00 · mise fixe 0,25 % · plafond quotidien 2 %.')
    st.info('Carnet privé à ton compte, séparé des anciens paris. Stockage local au serveur : sur '
            'Streamlit Cloud, un redéploiement peut le supprimer. Télécharger régulièrement la '
            'sauvegarde JSON ; elle peut être restaurée dans une bankroll vide. Aucun carnet n’est publié sur GitHub.')
    try:
        summary = ledger.state(db, owner)
    except ValueError:
        st.subheader('Créer la bankroll de simulation')
        with st.form(f'atp_initial_{owner}'):
            initial = st.number_input('Capital fictif initial (€)', 10.0, 1_000_000.0, 1000.0, 50.0)
            if st.form_submit_button('Initialiser la simulation'):
                try:
                    ledger.initialise(db, owner, initial)
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        backup = st.file_uploader('Ou restaurer une sauvegarde JSON de cette stratégie', type=['json'], key=f'atp_restore_{owner}')
        if backup is not None and st.button('Restaurer dans ce compte vide', key=f'atp_restore_button_{owner}'):
            try:
                ledger.restore_backup(db, owner, backup.getvalue().decode('utf-8'))
                st.rerun()
            except (ValueError, KeyError, TypeError) as error:
                st.error(f'Sauvegarde refusée : {error}')
        return

    cols = st.columns(5)
    for col, label, value in zip(cols, ['Bankroll simulée', 'Disponible', 'Mises ouvertes', 'Résultat net', 'ROI sur mises'],
                                 [f"{summary['balance_cents']/100:.2f} €", f"{summary['available_cents']/100:.2f} €",
                                  f"{summary['reserved_cents']/100:.2f} €", f"{summary['profit_cents']/100:+.2f} €",
                                  '—' if summary['roi'] is None else f"{summary['roi']:+.2%}"]):
        col.metric(label, value)
    st.caption(f"Budget encore disponible aujourd’hui : {summary['day_remaining_cents']/100:.2f} € ; "
               f"prochaine mise théorique au maximum {ledger.proposed_stake(summary)/100:.2f} €. "
               'Journée Europe/Paris ; le montant est revérifié à l’enregistrement, sans augmenter les mises avec les gains du jour.')
    analyse, journal = st.tabs(['Analyser un match', 'Carnet et sauvegarde'])
    with journal:
        _history(db, owner, summary)
    with analyse:
        metadata = root / 'models/tennis_strategy/metadata.json'
        try:
            fingerprint = engine.digest(metadata)
            bundle = _bundle(str(root), fingerprint)
            meta, history, _, _ = bundle
        except Exception as error:
            st.error(f'Modèle ATP indisponible : {type(error).__name__}. '
                     'Reconstruire le paquet avec scripts/prepare_tennis_strategy.py.')
            return
        evidence = meta['evidence']
        interval = evidence['uncertainty']['ci95']
        st.caption(f"Diagnostic historique 2023–2025 : {evidence['roi']['0.02']:+.2%} sur "
                   f"{evidence['settled']} paris réglés ; intervalle 95 % [{interval[0]:+.2%} ; {interval[1]:+.2%}]. "
                   'Filtre de validation non franchi. Les prix français constituent une nouvelle expérience, pas une reproduction prouvée de Bet365.')
        st.markdown(f"Historique disponible jusqu’au **{meta['history_last_date']}** ; modèle annuel **{meta['model_year']}**.")
        reasons = engine.freshness_reasons(meta)
        for reason in reasons:
            st.error(reason)
        if reasons:
            st.caption('Le carnet reste utilisable. La mise à jour télécharge les résultats, contrôle les données et conserve le modèle annuel ; aucune modification de la bankroll.')
            if st.button('Actualiser les données de la stratégie ATP', key=f'atp_refresh_{owner}'):
                with st.spinner('Téléchargement et vérification des résultats ATP…'):
                    result = run_task(root, 'tennis_strategy_refresh')
                if result['ok']:
                    _bundle.clear()
                    _score.clear()
                    st.rerun()
                else:
                    st.error('Actualisation non validée : les sélections restent bloquées. Aucun historique périmé ne sera autorisé.')
                    st.code(result['output'], language='text')
            with st.expander('Commandes pour une installation locale'):
                st.code('python3 scripts/refresh_tennis_strategy.py', language='bash')
            return
        if st.button('Consulter les cotes ATP françaises (consomme du quota API)', key=f'atp_prices_{owner}'):
            with st.spinner('Consultation explicite du fournisseur…'):
                catalogue = active_sports(root)
                quotes, errors = [], []
                if catalogue.ok:
                    keys = [s['key'] for s in catalogue.events
                            if str(s.get('key', '')).startswith('tennis_atp_') and s.get('active')]
                    for key in keys[:8]:
                        response = fetch_h2h_odds(root, key, regions='fr')
                        if response.ok:
                            quotes.extend(engine.french_quotes(response.events))
                        else:
                            errors.append(response.error)
                else:
                    errors.append(catalogue.error)
                st.session_state[f'atp_quotes_{owner}'] = quotes
                if errors:
                    st.warning(' ; '.join(errors))
                if not quotes:
                    st.info('Aucune paire de cotes ATP française récente obtenue. La saisie manuelle reste possible ; aucun prix n’est inventé.')
        quotes = st.session_state.get(f'atp_quotes_{owner}', [])
        selected_quote = None
        if quotes:
            st.dataframe(pd.DataFrame(quotes), hide_index=True)
            quote_index = st.selectbox('Relevé à analyser (identités et contexte à confirmer)', [-1, *range(len(quotes))],
                                      format_func=lambda i: 'Saisie manuelle' if i == -1 else
                                      f"{quotes[i]['player_1']} — {quotes[i]['player_2']} / {engine.BOOKMAKERS[quotes[i]['bookmaker']]}",
                                      key=f'atp_quote_choice_{owner}')
            if quote_index >= 0:
                selected_quote = quotes[quote_index]
        profiles = engine.latest_profiles(history, pd.Timestamp.now(tz='Europe/Paris'))
        players = sorted(profiles.index.tolist())
        if len(players) < 2:
            st.error('Historique de joueurs incomplet.')
            return
        context_key = selected_quote['event_id'] + selected_quote['bookmaker'] if selected_quote else 'manual'
        local = datetime.now(ZoneInfo('Europe/Paris'))
        default_start = engine.utc(selected_quote['start']).tz_convert('Europe/Paris').to_pydatetime() if selected_quote else local + timedelta(hours=2)
        st.caption('Confirmer les identités exactes, la surface, le tour et les classements actuels. '
                   'Le fournisseur de cotes ne fournit pas tous ces champs. Aucun rapprochement flou automatique.')
        with st.form(f'atp_fixture_{owner}_{context_key}'):
            c1, c2 = st.columns(2)
            with c1:
                left = st.selectbox('Joueur correspondant au côté 1', players)
                rank1 = st.number_input('Classement ATP actuel — côté 1', 1, 3000, 100)
                points1 = st.number_input('Points ATP actuels — côté 1', 0, 30000, 500)
                odds1 = st.number_input('Cote côté 1', 1.01, 100.0,
                                        float(selected_quote['odds_1']) if selected_quote else 2.0, .01,
                                        disabled=selected_quote is not None)
            with c2:
                right = st.selectbox('Joueur correspondant au côté 2', players, index=1)
                rank2 = st.number_input('Classement ATP actuel — côté 2', 1, 3000, 100)
                points2 = st.number_input('Points ATP actuels — côté 2', 0, 30000, 500)
                odds2 = st.number_input('Cote côté 2', 1.01, 100.0,
                                        float(selected_quote['odds_2']) if selected_quote else 2.0, .01,
                                        disabled=selected_quote is not None)
            tournament = st.selectbox('Tournoi (libellé historique)', sorted(history['tourney_name'].dropna().unique()))
            surface = st.selectbox('Surface confirmée', ['Hard', 'Clay', 'Grass'])
            indoor = st.selectbox('Conditions confirmées', ['Outdoor', 'Indoor'])
            round_name = st.selectbox('Tour', list(engine.ROUNDS), format_func=engine.ROUNDS.get)
            best_of = st.selectbox('Format', [3, 5], format_func=lambda n: f'Au meilleur des {n} sets')
            level = st.selectbox('Catégorie', ['250', '500', 'M', 'G', 'F', 'A'],
                                 format_func=lambda v: {'M':'Masters 1000', 'G':'Grand Chelem', 'F':'Finals', 'A':'Autre ATP principal'}.get(v, 'ATP '+v))
            book = st.selectbox('Bookmaker français', list(engine.BOOKMAKERS),
                                index=list(engine.BOOKMAKERS).index(selected_quote['bookmaker']) if selected_quote else 0,
                                format_func=engine.BOOKMAKERS.get, disabled=selected_quote is not None)
            day = st.date_input('Date de début (Paris)', default_start.date(), disabled=selected_quote is not None)
            clock = st.time_input('Heure de début (Paris)', default_start.time().replace(tzinfo=None), disabled=selected_quote is not None)
            confirmed = st.checkbox('Je confirme le simple ATP tableau principal, les deux identités dans cet ordre, '
                                    'le contexte et les classements ; les deux cotes sont disponibles chez ce même opérateur '
                                    '(consultées maintenant si saisie manuelle).')
            submitted = st.form_submit_button('Calculer selon la stratégie figée')
        if submitted:
            st.session_state.pop(f'atp_candidate_{owner}', None)
            fixture = {'tour':'ATP', 'singles_main_draw':True, 'player_1':left, 'player_2':right,
                       'player_1_rank':rank1, 'player_2_rank':rank2, 'player_1_rank_points':points1,
                       'player_2_rank_points':points2, 'odds_1':odds1, 'odds_2':odds2,
                       'tournament':tournament, 'surface':surface, 'indoor':indoor, 'round':round_name,
                       'best_of':best_of, 'level':level, 'bookmaker':book,
                       'start':datetime.combine(day, clock, tzinfo=ZoneInfo('Europe/Paris')).isoformat(),
                       'quote_at':selected_quote['quote_at'] if selected_quote else engine.utc().isoformat(),
                       'price_source':'api_user_confirmed_mapping' if selected_quote else 'manual_user_confirmed',
                       'api_pair':selected_quote}
            try:
                if not confirmed:
                    raise ValueError('Confirmer les identités, le contexte, les classements et les cotes avant le calcul.')
                with st.spinner('Calcul des statistiques antérieures et de la sélection — peut prendre quelques minutes…'):
                    candidate = _score(str(root), fingerprint, json.dumps(fixture, sort_keys=True))
                st.session_state[f'atp_candidate_{owner}'] = candidate
            except (ValueError, KeyError, OSError) as error:
                st.error(str(error))
        candidate = st.session_state.get(f'atp_candidate_{owner}')
        if candidate:
            st.write(f"Calcul enregistré : **{candidate['fixture']['player_1']} — {candidate['fixture']['player_2']}**, "
                     f"{candidate['fixture']['tournament']} / {candidate['fixture']['start']}.")
            st.caption('Ce résultat correspond aux champs soumis ci-dessus, pas à des modifications non recalculées.')
            st.dataframe(pd.DataFrame({'Joueur':[candidate['fixture']['player_1'], candidate['fixture']['player_2']],
                          'Probabilité modèle':candidate['probabilities'], 'EV estimée nette':candidate['expected_returns']}), hide_index=True)
            st.info(candidate['reason'])
            if candidate['eligible']:
                st.write(f"Sélection théorique : **{candidate['pick']}**, cote **{candidate['odds']:.2f}**.")
                if st.button('Enregistrer ce pari en simulation', key=f'atp_record_{owner}'):
                    try:
                        if candidate['model_sha256'] != meta['files']['booster.ubj'] or candidate['history_sha256'] != meta['files']['history.csv.gz']:
                            raise ValueError('Données ou modèle actualisés : recalcul obligatoire.')
                        amount = ledger.record(db, owner, candidate)
                        st.session_state.pop(f'atp_candidate_{owner}', None)
                        st.success(f'Simulation enregistrée : {amount:.2f} €. Consulter le carnet.')
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
