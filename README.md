# FlexiCrime
Predicting crime hotspots in a city is a complex and critical task with significant societal implications. The presence of numerous spatiotemporal correlations and irregularities poses substantial challenges to this endeavor. Existing methods commonly employ fixed time granularities and sequence prediction models. However, determining appropriate time granularities is difficult, leading to inaccurate predictions for specific time windows. For example, users might ask, “what are the crime hotspots during 12:00-20:00?” To address this issue, we introduce FlexiCrime, a novel event-centric framework for predicting crime hotspots with flexible time intervals. FlexiCrime incorporates a continuous-time attention network that captures correlations between crime events, which learns crime context features, representing general crime patterns across time points and locations. Furthermore, we introduce a type-aware spatiotemporal point process that learns crime-evolving features, measuring the risk of specific crime types at a given time and location by considering the frequency of past crime events. The crime context and evolving features together allow us to predict whether an urban area is a crime hotspot given a future time interval. To evaluate FlexiCrime’s effectiveness, we conducted experiments using real-world datasets from two cities, covering twelve crime types. The results show that our model outperforms baseline techniques in predicting crime hotspots over flexible time intervals.

# Guidance
+ model_v2: Continuous-time Attention Network for Event Aggregation.
+ neural_stpp: Type-aware Spatiotemporal Point Process for Event Evolution.
+ experiment_v2: Experiments of the performance of FlexiCrime.

# Step:

1. Init: run setup.py
2. Train Continuous-time Attention Network: run `experiment_v2/NYC/train.py` and `experiment_v2/SEA/train.py`
3. Construct data for Type-aware Spatiotemporal Point Process: run `neural_stpp/zNYC_generate_crime*.py` and `neural_stpp/zSEA_generate_crime*.py`
4. Train Type-aware Spatiotemporal Point Process: run `neural_stpp/train_stpp.py`
5. Retrain Continuous-time Attention Network: run `experiment_v2/NYC/train_fine_tuning_2.py` and `experiment_v2/SEA/train_fine_tuning_2.py`
