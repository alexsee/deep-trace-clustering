import numpy as np

class Evaluation(object):
    def __init__(self,
                event_log):
        self._event_log = event_log
    
    def evaluate_clusters(self, n_clusters, pred_labels):
        pred_labels = np.asarray(pred_labels)
        
        clusters = []
        for i in range(n_clusters):
            clusters.append([x for index, x in enumerate(self._event_log._event_log) if pred_labels[index] == i])
            
        from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
        from pm4py.evaluation.precision import evaluator as precision_evaluator
        from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator

        from pm4py.algo.discovery.inductive import factory as inductive_miner
        from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

        from pm4py.evaluation.precision import factory as precision_factory

        avgFitness = 0
        avgPrecision = 0
        avgSimp = 0

        for cluster in clusters:
            inductive_petri, inductive_initial_marking, inductive_final_marking = heuristics_miner.apply(cluster, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})

            fitness_inductive = replay_fitness_evaluator.apply(cluster, inductive_petri, inductive_initial_marking, inductive_final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
            precision_inductive = precision_evaluator.apply(cluster, inductive_petri, inductive_initial_marking, inductive_final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
            
            simp = simplicity_evaluator.apply(inductive_petri)
            
            avgFitness += fitness_inductive['average_trace_fitness'] * len(cluster)
            avgPrecision += precision_inductive * len(cluster)
            avgSimp += simp * len(cluster)

        avgFitness /= len(self._event_log._event_log)
        avgPrecision /= len(self._event_log._event_log)
        avgSimp /= len(self._event_log._event_log)

        return avgFitness, avgPrecision, avgSimp