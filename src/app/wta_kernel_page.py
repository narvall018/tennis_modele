"""WTA paper scanning: explicit API call, confirmed surface, isolated journal."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app import wta_kernel_strategy as engine
from src.app import wta_kernel_ledger as ledger
from src.app.odds_api import active_sports, fetch_h2h_odds
from src.app.tennis_strategy_page import _bankroll_metrics, _history


@st.cache_resource(show_spinner=False, max_entries=2)
def _bundle(root, fingerprint):
    return engine.load_bundle(Path(root))


def collect(root):
    result = {'events': [], 'errors': [], 'competitions': 0, 'omitted': 0,
              'at': engine.utc().isoformat(), 'remaining': None}
    try:
        catalogue = active_sports(root)
        if not catalogue.ok:
            result['errors'].append('Catalogue WTA indisponible.'); return result
        keys = sorted({s['key'] for s in catalogue.events if s.get('active')
            and str(s.get('key', '')).startswith('tennis_wta_') and not s.get('has_outrights')})
        result['competitions'], result['omitted'] = len(keys), max(0, len(keys)-8)
        for key in keys[:8]:
            response = fetch_h2h_odds(root, key, regions='fr')
            result['remaining'] = response.remaining
            if not response.ok:
                result['errors'].append(f'{key} : cotes indisponibles.'); continue
            result['events'].extend(e for e in response.events if isinstance(e, dict) and e.get('sport_key') == key)
    except Exception as error:
        result['errors'].append(f'Consultation interrompue ({type(error).__name__}).')
    return result


def analyse(bundle, events, contexts, now=None):
    quotes = engine.quotes(events, now)
    results, candidates = [], {}
    for quote in quotes:
        context = contexts.get(quote['sport_key'], {})
        display = {'Match': quote['player_1']+' — '+quote['player_2'],
                   'Bookmaker': engine.BOOKMAKERS[quote['bookmaker']], 'Tournoi': quote['competition']}
        try:
            fixture = {**quote, 'api_pair': quote, 'tour': 'WTA', 'tournament': quote['competition'],
                       'surface': context.get('surface'), 'singles_main_draw': context.get('confirmed') is True}
            candidate = engine.score_fixture(bundle, fixture, now)
            display.update(Statut='Signal expérimental' if candidate['eligible'] else 'Pas de signal',
                           Motif=candidate['reason'], EV_max=max(candidate['expected_returns']))
            if candidate['eligible']:
                old = candidates.get(candidate['event_key'])
                if old is None or candidate['expected_returns'][candidate['selected_side']] > old['expected_returns'][old['selected_side']]:
                    candidates[candidate['event_key']] = candidate
        except (ValueError, KeyError, TypeError) as error:
            display.update(Statut='Non analysable', Motif=str(error))
        results.append(display)
    return results, sorted(candidates.values(), key=lambda c: (c['fixture']['start'], c['event_key']))


@st.fragment(run_every='30s')
def _results(root, owner, snapshot, contexts):
    db = Path(root)/'bets/wta_kernel_strategy.sqlite3'
    if not pd.Timedelta(0) <= engine.utc()-engine.utc(snapshot['at']) <= pd.Timedelta(minutes=5):
        st.warning('Relevé expiré : consulter à nouveau les cotes.'); return
    try:
        fingerprint = engine.digest(Path(root)/engine.FOLDER/'metadata.json')
        bundle = _bundle(root, fingerprint)
        results, candidates = analyse(bundle, snapshot['events'], contexts)
    except (ValueError, OSError, KeyError) as error:
        st.error(f'Analyse bloquée : {error}'); return
    if not results:
        st.info('Aucune paire française récente et admissible. Les matchs manquants ne sont pas évalués.'); return
    analysed = sum(r['Statut'] != 'Non analysable' for r in results)
    st.write(f'{analysed} paire(s) analysée(s), {len(results)-analysed} bloquée(s), '
             f'{len(candidates)} match(s) avec signal théorique.')
    with st.expander('Résultats et motifs par bookmaker', expanded=not bool(candidates)):
        st.dataframe(pd.DataFrame(results), hide_index=True)
    if not candidates:
        st.info('Aucune opportunité selon la règle parmi les matchs analysables.' if analysed
                else 'Analyse impossible : confirmer le contexte et vérifier les motifs.'); return
    summary = ledger.state(db, owner)
    existing = {b['event_key'] for b in summary['bets']}
    candidates = [c for c in candidates if c['event_key'] not in existing]
    budget = min(summary['day_remaining_cents'], summary['available_cents'])
    fixed = ledger.proposed_stake(summary); planned = []
    for c in candidates:
        stake = min(fixed, budget); budget -= stake
        planned.append({'Joueuse': c['pick'], 'Bookmaker': engine.BOOKMAKERS[c['fixture']['bookmaker']],
                        'Cote': c['odds'], 'Probabilité estimée': c['probability'],
                        'EV nette estimée': c['expected_returns'][c['selected_side']], 'Mise simulée €': stake/100})
    if not planned:
        st.info('Ces matchs figurent déjà dans ton carnet.'); return
    st.dataframe(pd.DataFrame(planned), hide_index=True)
    st.caption('Une seule sélection par match, parmi les bookmakers analysés. Prix français et choix '
               'multi-bookmakers : expérience prospective distincte du backtest Bet365. Budget partagé entre les propositions ci-dessus.')
    if planned[0]['Mise simulée €'] <= 0:
        st.info('Budget ou bankroll disponible épuisé.'); return
    if st.button('Enregistrer la sélection prioritaire en simulation', key=f'wta_kernel_record_{owner}'):
        try:
            amount = ledger.record(db, owner, candidates[0], root=Path(root))
            st.success(f'{amount:.2f} € enregistrés en simulation.'); st.rerun()
        except (ValueError, KeyError, OSError) as error: st.error(str(error))


def render_wta_kernel_page(root: Path, user_id: int, username: str):
    owner = f'{int(user_id)}:{username}'; db = root/'bets/wta_kernel_strategy.sqlite3'
    st.title('Stratégie WTA — service/retour et noyau')
    st.warning('Rentabilité non démontrée. Modèle choisi après exploration ; filtre de validation non franchi. '
               'Simulation uniquement, aucun pari transmis à un bookmaker.')
    st.caption('EV estimée ≥ 3 % après décote de 2 % des gains · cotes 1,30–5,00 · '
               'mise 0,25 % du capital du début de journée · plafond 2 % par jour, Europe/Paris.')
    st.info('Nouveau carnet privé, séparé de l’ATP et de l’ancienne WTA. Le stockage serveur peut disparaître '
            'au redéploiement : télécharger régulièrement la sauvegarde JSON. Les anciens carnets restent dans les archives ATP.')
    try: summary = ledger.state(db, owner)
    except ValueError:
        with st.form(f'wta_kernel_initial_{owner}'):
            amount = st.number_input('Capital fictif initial (€)', 10., 1_000_000., 1000., 50.)
            if st.form_submit_button('Initialiser la simulation'):
                ledger.initialise(db, owner, amount); st.rerun()
        backup = st.file_uploader('Restaurer une sauvegarde WTA kernel', type=['json'], key=f'wta_kernel_restore_{owner}')
        if backup is not None and st.button('Restaurer dans ce compte vide', key=f'wta_kernel_restore_button_{owner}'):
            try:
                ledger.restore_backup(db, owner, backup.getvalue().decode('utf-8')); st.rerun()
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error: st.error(str(error))
        return
    _bankroll_metrics(summary, ledger_api=ledger)
    scan, journal = st.tabs(['Opportunités WTA', 'Carnet et sauvegarde'])
    with journal: _history(db, owner, summary, ledger_api=ledger, prefix='wta_kernel')
    with scan:
        try:
            fingerprint = engine.digest(root/engine.FOLDER/'metadata.json')
            meta, _, _ = _bundle(str(root), fingerprint)
        except (OSError, ValueError, KeyError):
            st.error('Paquet WTA kernel indisponible ou incohérent.')
            st.code('python3 scripts/prepare_wta_kernel.py'); return
        evidence = meta['evidence']; lower, upper = evidence['uncertainty']['ci95']
        st.caption(f"Diagnostic 2023–2025 : {evidence['roi']['0.02']:+.2%} sur {evidence['settled']} paris réglés "
                   f"(rendement normalisé) ; intervalle à 95 % [{lower:+.2%} ; {upper:+.2%}]. "
                   '2021–2022 : sélection non validée. Prix français non validés.')
        st.caption(f"Modèle {meta['model_year']} ; entraînement jusqu’au {meta['training_max_date']} ; "
                   f"début de tournoi le plus récent dans les statistiques : {meta['history_last_date']}. "
                   'Les statistiques ne sont utilisées qu’après 28 jours, avec au moins cinq matchs par joueuse.')
        for reason in engine.freshness_reasons(meta): st.error(reason)
        if st.button('Actualiser les statistiques WTA', key=f'wta_kernel_refresh_{owner}'):
            try:
                from src.app.wta_kernel_refresh import refresh
                with st.spinner('Actualisation des statistiques publiques, sans réentraînement…'): refresh(root)
                _bundle.clear(); st.rerun()
            except Exception as error: st.error(f'Actualisation refusée ({type(error).__name__}) ; ancien modèle et carnet conservés.')
        if engine.freshness_reasons(meta):
            st.code('python3 scripts/refresh_wta_kernel.py'); return
        st.caption('Pas de référence Bet365/Pinnacle, ni de classement à saisir. L’API fournit les cotes et les noms, '
                   'mais pas une surface ni un statut de tableau suffisamment fiables : confirmer ces deux éléments par compétition.')
        if st.button('Consulter les cotes WTA françaises (consomme du quota API)', key=f'wta_kernel_fetch_{owner}'):
            with st.spinner('Consultation du fournisseur…'): snapshot = collect(root)
            st.session_state[f'wta_kernel_snapshot_{owner}'] = snapshot
            st.session_state.pop(f'wta_kernel_contexts_{owner}', None)
        snapshot = st.session_state.get(f'wta_kernel_snapshot_{owner}')
        if snapshot:
            for error in snapshot['errors']: st.warning(error)
            st.caption(f"{snapshot['competitions']} compétition(s), {len(snapshot['events'])} match(s) reçu(s). "
                       f"Quota restant : {snapshot['remaining']}. Compétitions non interrogées : {snapshot['omitted']}.")
            if not snapshot['events']:
                st.info('Aucun match WTA reçu. Aucune opportunité ne peut être évaluée.'); return
            competitions = {e['sport_key']: e.get('sport_title', e['sport_key']) for e in snapshot['events']}
            with st.form(f'wta_kernel_context_{owner}'):
                contexts = {}
                for key, label in sorted(competitions.items()):
                    st.write(label)
                    surface = st.selectbox('Surface confirmée', ['Hard', 'Clay', 'Grass'], index=None, key=f'wta_kernel_surface_{owner}_{key}')
                    confirmed = st.checkbox('Je confirme : simple WTA tableau principal, pas qualifications ni WTA125.', key=f'wta_kernel_confirm_{owner}_{key}')
                    contexts[key] = {'surface': surface, 'confirmed': confirmed}
                if st.form_submit_button('Analyser automatiquement les matchs et les mises'):
                    st.session_state[f'wta_kernel_contexts_{owner}'] = contexts
            saved = st.session_state.get(f'wta_kernel_contexts_{owner}')
            if saved is not None:
                st.caption('Analyse du dernier contexte soumis. Fraîcheur revérifiée toutes les 30 secondes, sans nouvel appel API.')
                _results(str(root), owner, snapshot, saved)
