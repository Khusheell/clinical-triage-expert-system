% triage_kb.pl

:- discontiguous heuristic/3.
:- discontiguous node/3.
:- discontiguous edge/2.
:- discontiguous total_diseases/2.
:- discontiguous disease_symptoms/2.


% DYNAMIC PREDICATES

:- dynamic(case_log/5).
:- dynamic(log_case/5).
:- dynamic(patient_symptom/1).
:- dynamic(patient_risk/1).


% FACTS
% (Your facts: symptom/2, etc. remain unchanged)
symptom(heart_attack, chest_pain).
symptom(heart_attack, shortness_of_breath).
symptom(heart_attack, arm_pain).
symptom(heart_attack, nausea).
symptom(stroke, sudden_weakness).
symptom(stroke, confusion).
symptom(stroke, vision_loss).
symptom(infection, fever).
symptom(infection, cough).
symptom(infection, sore_throat).
symptom(infection, body_aches).
symptom(common_cold, fever).
symptom(common_cold, cough).
symptom(common_cold, runny_nose).
symptom(common_cold, sore_throat).
symptom(covid_19, fever).
symptom(covid_19, cough).
symptom(covid_19, loss_of_taste).
symptom(covid_19, shortness_of_breath).


% CRITICAL SYMPTOMS
% (Unchanged)
critical_symptom(uncontrolled_bleeding).
critical_symptom(sudden_weakness).
critical_symptom(chest_pain).
critical_symptom(severe_shortness_of_breath).


% SYMPTOM SCORING
% (Unchanged)
symptom_score(chest_pain, 25, 10).
symptom_score(shortness_of_breath, 15, 10).
symptom_score(severe_shortness_of_breath, 25, 10).
symptom_score(fever, 5, 5).
symptom_score(cough, 3, 2).
symptom_score(arm_pain, 10, 5).
symptom_score(nausea, 5, 0).
symptom_score(sudden_weakness, 30, 0).
symptom_score(confusion, 20, 0).
symptom_score(vision_loss, 25, 0).
symptom_score(body_aches, 5, 0).
symptom_score(runny_nose, 1, 0).
symptom_score(sore_throat, 2, 0).
symptom_score(loss_of_taste, 10, 0).
symptom_score(uncontrolled_bleeding, 50, 0).
symptom_score(_, 0, 0).


% RECURSION / LIST OPS / ARITHMETIC
% (Unchanged)
calculate_score([], _, 0, []).
calculate_score([Sym|Rest], Risks, Total, [break(Sym, Base, RiskAdj)|RestB]) :-
    symptom_score(Sym, Base, Adj),
    ( (member(Sym, [chest_pain, shortness_of_breath]), member(smoker, Risks))
      -> RiskAdj = Adj ; RiskAdj = 0 ),
    Score is Base + RiskAdj,
    calculate_score(Rest, Risks, RestTotal, RestB),
    Total is Score + RestTotal.


% === MODIFIED TRIAGE RULES ===
% (This is the critical change)
% (Now find_triage_level/6, with Score as the last argument)

find_triage_level(Symptoms, _, 'Emergency', 'Critical symptom detected', [], 100) :-
    member(S, Symptoms),
    critical_symptom(S), !.

find_triage_level(Symptoms, Risks, 'Emergency', 'Score exceeds emergency threshold', Breakdown, Score) :-
    calculate_score(Symptoms, Risks, Score, Breakdown),
    Score >= 40, !.

find_triage_level(Symptoms, Risks, 'Urgent', 'High score but non-critical', Breakdown, Score) :-
    calculate_score(Symptoms, Risks, Score, Breakdown),
    Score >= 15,
    \+ (member(S, Symptoms), critical_symptom(S)),
    \+ (Score >= 40),
    !.

find_triage_level(Symptoms, Risks, 'Routine', 'Low risk - routine care', Breakdown, Score) :-
    calculate_score(Symptoms, Risks, Score, Breakdown).

% === END OF MODIFIED RULES ===


% (Rest of your file: QUANTIFIER, A*, AO*, etc. remains unchanged)

% QUANTIFIER EXAMPLE
hallmark_symptoms(infection, [fever, cough, sore_throat]).
has_all_infection_symptoms(P) :-
    hallmark_symptoms(infection, L),
    forall(member(S, L), member(S, P)).

% A* SEARCH FACTS
action(initial_assessment, check_vitals, check_vitals, 1).
action(initial_assessment, patient_history, patient_history, 1).
action(check_vitals, stabilize, stabilize_patient, 5).
action(check_vitals, consult_specialist, consult_specialist, 3).
action(check_vitals, discharge, discharge, 2).
action(patient_history, consult_specialist, consult_specialist, 3).
action(patient_history, diagnostic_tests, diagnostic_tests, 4).
action(consult_specialist, diagnostic_tests, diagnostic_tests, 2).
action(consult_specialist, admit_ward, admit_ward, 5).
action(diagnostic_tests, admit_ward, admit_ward, 3).
action(diagnostic_tests, review_results, review_results, 2).
action(review_results, admit_ward, admit_ward, 3).
action(review_results, discharge, discharge, 2).
action(stabilize, admit_icu, admit_icu, 10).
action(admit_ward, discharge, discharge, 4).
action(admit_icu, discharge, discharge, 8).

triage_goal('Emergency', admit_icu).
triage_goal('Urgent', admit_ward).
triage_goal('Routine', discharge).

heuristic(initial_assessment, admit_to_icu, 10) :- !.
heuristic(check_vitals, admit_to_icu, 8) :- !.
heuristic(patient_history, admit_to_icu, 9) :- !.
heuristic(stabilize, admit_to_icu, 1) :- !.
heuristic(consult_specialist, admit_to_icu, 5) :- !.
heuristic(diagnostic_tests, admit_to_icu, 6) :- !.
heuristic(review_results, admit_to_icu, 7) :- !.
heuristic(admit_ward, admit_to_icu, 3) :- !.
heuristic(admit_icu, admit_to_icu, 0) :- !.
heuristic(discharge, admit_to_icu, 100) :- !.
heuristic(initial_assessment, admit_ward, 6) :- !.
heuristic(check_vitals, admit_ward, 5) :- !.
heuristic(patient_history, admit_ward, 4) :- !.
heuristic(stabilize, admit_ward, 8) :- !.
heuristic(consult_specialist, admit_ward, 2) :- !.
heuristic(diagnostic_tests, admit_ward, 1) :- !.
heuristic(review_results, admit_ward, 1) :- !.
heuristic(admit_ward, admit_ward, 0) :- !.
heuristic(admit_icu, admit_ward, 3) :- !.
heuristic(discharge, admit_ward, 100) :- !.
heuristic(initial_assessment, discharge, 3) :- !.
heuristic(check_vitals, discharge, 2) :- !.
heuristic(patient_history, discharge, 3) :- !.
heuristic(stabilize, discharge, 10) :- !.
heuristic(consult_specialist, discharge, 4) :- !.
heuristic(diagnostic_tests, discharge, 2) :- !.
heuristic(review_results, discharge, 1) :- !.
heuristic(admit_ward, discharge, 1) :- !.
heuristic(admit_icu, discharge, 5) :- !.
heuristic(discharge, discharge, 0) :- !.
heuristic(_, _, 1).

% AO* GRAPH
node(respiratory_distress, or, 1).
edge(respiratory_distress, [covid_19_diagnosis]).
edge(respiratory_distress, [infection_diagnosis]).
edge(respiratory_distress, [cardiac_event_diagnosis]).
node(cardiac_event_diagnosis, or, 1).
edge(cardiac_event_diagnosis, [heart_attack_diagnosis]).
edge(cardiac_event_diagnosis, [stroke_diagnosis]).
node(infection_diagnosis, or, 1).
edge(infection_diagnosis, [covid_19_diagnosis]).
edge(infection_diagnosis, [infection_diagnosis_simple]).
node(heart_attack_diagnosis, and, 1).
edge(heart_attack_diagnosis, [check_troponin, check_ecg]).
node(stroke_diagnosis, and, 1).
edge(stroke_diagnosis, [check_ct_scan, check_neurological_exam]).
node(covid_19_diagnosis, and, 1).
edge(covid_19_diagnosis, [check_pcr_test, check_loss_of_taste]).
node(infection_diagnosis_simple, and, 1).
edge(infection_diagnosis_simple, [check_flu_test, check_body_aches]).

ao_heuristic(check_troponin, 10).
ao_heuristic(check_ecg, 5).
ao_heuristic(check_ct_scan, 20).
ao_heuristic(check_neurological_exam, 5).
ao_heuristic(check_pcr_test, 15).
ao_heuristic(check_loss_of_taste, 1).
ao_heuristic(check_flu_test, 10).
ao_heuristic(check_body_aches, 1).
ao_heuristic(check_runny_nose, 1).
ao_heuristic(check_sore_throat, 1).
ao_heuristic(check_blood_sugar, 5).

% FIND HYPOTHESIS
find_hypothesis(Symptoms, respiratory_distress) :- member(shortness_of_breath, Symptoms), !.
find_hypothesis(Symptoms, cardiac_event_diagnosis) :- member(chest_pain, Symptoms), !.
find_hypothesis(Symptoms, infection_diagnosis) :- member(fever, Symptoms), !.
find_hypothesis(Symptoms, stroke_diagnosis) :- member(sudden_weakness, Symptoms), !.
find_hypothesis(_, infection_diagnosis).

% TOTAL DISEASES
total_diseases(TermAtom, Total) :-
    findall(D, symptom(D, TermAtom), Ds),
    Ds \= [],
    sort(Ds, UniqueDs),
    length(UniqueDs, Total), !.
total_diseases(TermAtom, Total) :-
    findall(S, symptom(TermAtom, S), Ss),
    Ss \= [],
    sort(Ss, UniqueSs),
    length(UniqueSs, Total), !.
total_diseases(_, 0).

disease_symptoms(D, S) :- symptom(D, S).

% CASE LOG HELPERS
get_all_cases(List) :-
    findall(case_log(ID, Time, Level, Symptoms, Risks), case_log(ID, Time, Level, Symptoms, Risks), List).

latest_case(case_log(ID, Time, Level, Symptoms, Risks)) :-
    findall((T, ID0, Level0, Symptoms0, Risks0), case_log(ID0, T, Level0, Symptoms0, Risks0), Pairs),
    Pairs \= [],
    last(Pairs, (_, ID, Level, Symptoms, Risks)),
    time_string(Time),
    !.

% A* SEARCH IMPLEMENTATION (Now with correct, unique, step-by-step logging)

% a_star/5 predicate:
% a_star(Start, Goal, Path, Cost, Log)
% Log will be a list of nodes visited in order of exploration.

a_star(Start, Goal, Path, Cost, Log) :-
    % Start the search. The log will be built in the helper.
    astar_search([(0, 0, [Start])], Goal, RevPath, Cost, [], RevLog), % Start with an empty log
    reverse(RevPath, Path),
    reverse(RevLog, Log). % Reverse the final log

% Base case: Goal is found.
% Add the Goal to the log and return the accumulated log.
astar_search([(_, Cost, [Goal|RestPath])|_], Goal, [Goal|RestPath], Cost, LogAccum, [Goal|LogAccum]).

% Recursive case:
astar_search([(_, G, [Current|RestPath])|RestQueue], Goal, FinalPath, FinalCost, LogAccum, FinalLog) :-
    
    % Check if we've *already* logged this node.
    ( member(Current, LogAccum) ->
        NewLogAccum = LogAccum % Don't add it again
    ;
        NewLogAccum = [Current|LogAccum] % Add it to the front (building reverse log)
    ),
    
    findall(
        (FNew, GNew, [Next,Current|RestPath]),
        (
            action(Current, Next, _, StepCost),
            \+ member(Next, [Current|RestPath]), % Avoid cycles in the *path*
            GNew is G + StepCost,
            heuristic(Next, Goal, H),
            FNew is GNew + H
        ),
        NewPaths
    ),
    append(RestQueue, NewPaths, TempQueue),
    sort(TempQueue, SortedQueue),
    
    % Recurse with the new log
    astar_search(SortedQueue, Goal, FinalPath, FinalCost, NewLogAccum, FinalLog).

% Fallback for when the queue is empty (no path found)
astar_search([], _, [], 0, LogAccum, LogAccum).

% AO* SEARCH IMPLEMENTATION
:- dynamic(ao_solved/3).

ao_star(Root, Solution, Cost, Steps) :-
    retractall(ao_solved(_, _, _)),
    ao_expand(Root, Solution, Cost, Steps).

ao_expand(Node, [Node], Cost, [Node]) :-
    ao_heuristic(Node, Cost), !.

ao_expand(Node, [Node|Sol], Cost, [Node|Steps]) :-
    node(Node, or, _),
    edge(Node, [Child]),
    ao_expand(Child, Sol, SubCost, SubSteps),
    Cost is SubCost + 1,
    Steps = SubSteps.

ao_expand(Node, [Node|Sol], Cost, [Node|Steps]) :-
    node(Node, and, _),
    edge(Node, Children),
    findall(SubSol, (member(C, Children), ao_expand(C, SubSol, _, _)), SubSols),
    flatten(SubSols, Sol),
    findall(SubC, (member(C, Children), ao_expand(C, _, SubC, _)), SubCosts),
    sum_list(SubCosts, SumC),
    Cost is SumC + 1,
    Steps = Sol.

ao_expand(Node, [Node], 1, [Node]).