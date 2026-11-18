"""Plain-English descriptions of the five models (no numbers: every number on the pages comes from app/assets)."""

DIAGRAM_STYLE = ('bgcolor="transparent"; node [shape=box, style="rounded,filled", color="#c3c2b7", '
                 'fillcolor="#f2f1ec", fontname="Helvetica", fontsize=11]; edge [color="#898781", fontsize=10, '
                 'fontname="Helvetica"];')

MODELS = {
    "Logistic regression": {
        "family": "Linear model",
        "how": "Gives every input a weight, adds up *weight × value* for the applicant and squeezes the total through "
               "an S-shaped (logistic) curve into a score between 0 and 1. Training finds the weights that best "
               "separate defaulters from good applicants.",
        "analogy": "A credit-scorecard: points for each characteristic, added up into one total.",
        "why": "The industry-standard credit scorecard and the baseline every other model must beat. Each weight can "
               "be read and explained to a customer or a regulator.",
        "preprocessing": "Median imputation plus a 0/1 *missing* indicator for every numeric column, StandardScaler "
                         "(weights are only comparable on a common scale), one-hot encoding of categories (levels "
                         "under 1% pooled).",
        "imbalance": "`class_weight='balanced'`: each default is weighted up by the ratio of repaid loans to defaults, "
                     "so both classes count equally during training.",
        "tuning": "Not tuned: L2 regularisation with sklearn defaults, `max_iter=2000` so the solver converges.",
        "strengths": ["Fully transparent and auditable", "Fast and stable (smallest fold-to-fold spread)",
                      "Hardly overfits"],
        "weaknesses": ["Only straight-line relationships unless interactions are engineered by hand",
                       "Needs imputation, scaling and one-hot encoding first"],
        "interpretability": "Very high: one weight per input",
        "scaling": "Yes", "missing": "No: median imputation + missing indicators",
        "best_use": "Regulated decisions that must be explained line by line; a benchmark",
        "dot": "digraph { rankdir=LR; " + DIAGRAM_STYLE + ' a [label="Inputs\\n(age, income, scores, ...)"]; '
               'b [label="Weighted sum\\nw1·x1 + w2·x2 + ..."]; c [label="Logistic curve\\n1 / (1 + e^-z)"]; '
               'd [label="Risk score\\n0 - 1", fillcolor="#2a78d6", fontcolor="white"]; a -> b -> c -> d; }',
    },
    "Decision tree": {
        "family": "Single tree",
        "how": "Asks a sequence of yes/no questions about one feature at a time (\"is the average external score "
               "below a cut-off?\"), each time picking the question that best separates defaulters from good "
               "applicants. An applicant's score is the class-weighted share of defaulters in the final box (leaf) "
               "they land in.",
        "analogy": "A flow chart a loan officer could follow on paper.",
        "why": "The most interpretable non-linear model; it shows what a single tree can and cannot do, and it is the "
               "building block of the random forest and of boosting.",
        "preprocessing": "Median imputation and one-hot encoding (sklearn trees need numbers); no scaling, because a "
                         "split only compares one feature with a cut-off.",
        "imbalance": "`class_weight='balanced'` when choosing splits.",
        "tuning": "Grid search with 3-fold cross-validation over the maximum depth and the minimum number of "
                  "applicants per leaf.",
        "strengths": ["Can be drawn and read", "Captures thresholds and interactions automatically",
                      "Very fast to train and score"],
        "weaknesses": ["Unstable: small data changes can change the whole tree",
                       "Few distinct scores (one per leaf), so ranking is coarse",
                       "Deep trees overfit quickly"],
        "interpretability": "High while shallow",
        "scaling": "No", "missing": "No: median imputation",
        "best_use": "Explaining the main risk drivers as simple rules; a teaching model",
        "dot": "digraph { rankdir=TB; " + DIAGRAM_STYLE + ' q1 [label="Average external score low?"]; '
               'q2 [label="Score very low?"]; q3 [label="Late on past instalments?"]; '
               'l1 [label="High risk", fillcolor="#d03b3b", fontcolor="white"]; l2 [label="Medium risk"]; '
               'l3 [label="Medium risk"]; l4 [label="Low risk", fillcolor="#0ca30c", fontcolor="white"]; '
               'q1 -> q2 [label="yes"]; q1 -> q3 [label="no"]; q2 -> l1 [label="yes"]; q2 -> l2 [label="no"]; '
               'q3 -> l3 [label="yes"]; q3 -> l4 [label="no"]; }',
    },
    "Random forest": {
        "family": "Bagging of trees",
        "how": "Grows hundreds of deep decision trees, each on a random half of the applicants and choosing every "
               "split from a random subset of features, then averages their votes. Each tree overfits in its own "
               "way; averaging cancels much of that noise.",
        "analogy": "Asking hundreds of independent experts who each saw part of the evidence, then taking the "
                   "average opinion.",
        "why": "Represents *bagging*: shows how averaging many high-variance trees reduces variance, with no scaling "
               "and little tuning needed.",
        "preprocessing": "Same as the decision tree (median imputation, one-hot encoding).",
        "imbalance": "`class_weight='balanced_subsample'`: classes are re-weighted inside each tree's own sample.",
        "tuning": "Time-bounded random search with 3-fold cross-validation over the number of trees, maximum depth, "
                  "minimum leaf size and the share of features tried at each split.",
        "strengths": ["Robust, few settings matter much", "Captures interactions",
                      "Scores are the least inflated of the class-weighted models"],
        "weaknesses": ["Large model file and the slowest to train here", "Hard to explain a single decision",
                       "Memorises the training data (very high in-sample AUC)"],
        "interpretability": "Low (hundreds of trees); needs SHAP-style tools",
        "scaling": "No", "missing": "No: median imputation",
        "best_use": "A strong, low-maintenance baseline for tabular data",
        "dot": "digraph { rankdir=LR; " + DIAGRAM_STYLE + ' d [label="Training\\napplicants"]; '
               's1 [label="Random half 1"]; s2 [label="Random half 2"]; s3 [label="Random half N"]; '
               't1 [label="Tree 1"]; t2 [label="Tree 2"]; t3 [label="Tree N"]; '
               'v [label="Average of\\nthe votes", fillcolor="#eda100"]; d -> s1 -> t1 -> v; d -> s2 -> t2 -> v; '
               'd -> s3 -> t3 -> v; }',
    },
    "XGBoost": {
        "family": "Gradient boosting of trees",
        "how": "Builds shallow trees one after another. Each new tree is fitted to the mistakes the current "
               "ensemble still makes, and its predictions are added with a small learning rate. Training stops when "
               "a held-back validation set stops improving (early stopping).",
        "analogy": "A team of students where each one studies only the questions the previous ones got wrong.",
        "why": "A second, widely used boosting library: a robustness check on LightGBM with fixed, untuned settings.",
        "preprocessing": "None needed: missing values and categorical features are handled natively.",
        "imbalance": "`scale_pos_weight` = repaid applicants ÷ defaulters.",
        "tuning": "Not tuned (fixed settings from the original analysis); early stopping chooses the number of trees.",
        "strengths": ["High accuracy on tabular data", "Native missing-value handling",
                      "Regularisation built in"],
        "weaknesses": ["Many settings to tune", "Less transparent than a scorecard"],
        "interpretability": "Low; explained with SHAP",
        "scaling": "No", "missing": "Yes, natively",
        "best_use": "High-accuracy scoring when tuning time is available",
        "dot": "digraph { rankdir=LR; " + DIAGRAM_STYLE + ' t1 [label="Tree 1"]; e1 [label="Errors left"]; '
               't2 [label="Tree 2 fixes\\nthose errors"]; e2 [label="Errors left"]; t3 [label="... Tree N"]; '
               's [label="Sum of all trees\\n= risk score", fillcolor="#1baf7a", fontcolor="white"]; '
               't1 -> e1 -> t2 -> e2 -> t3 -> s; }',
    },
    "LightGBM": {
        "family": "Gradient boosting of trees",
        "how": "Gradient boosting like XGBoost, but trees grow leaf by leaf (always splitting the most useful leaf) "
               "and features are bucketed into histograms, which makes training very fast on hundreds of thousands "
               "of applicants. Small trees, a low learning rate, feature/row sampling and L2 regularisation keep it "
               "from overfitting; early stopping picks the number of trees.",
        "analogy": "The same team of students as XGBoost, studying faster and more carefully.",
        "why": "State of the art for large tabular data and the main model: native missing values and categories, "
               "fast enough for a proper random search with 5-fold cross-validation.",
        "preprocessing": "None needed: missing values and categorical features are handled natively.",
        "imbalance": "`scale_pos_weight` = repaid applicants ÷ defaulters.",
        "tuning": "Random search with 5-fold cross-validation over the number of leaves, learning rate, minimum "
                  "leaf size, feature and row sampling and L2 regularisation.",
        "strengths": ["Best ranking and decisions in this project", "Fast training",
                      "Native missing values and categories"],
        "weaknesses": ["Needs care against overfitting (regularisation, early stopping)",
                       "A single decision needs SHAP to explain"],
        "interpretability": "Low on its own; high with SHAP reason codes",
        "scaling": "No", "missing": "Yes, natively",
        "best_use": "The production scoring model, with SHAP reason codes for every decision",
        "dot": "digraph { rankdir=LR; " + DIAGRAM_STYLE + ' t1 [label="Small tree 1"]; '
               't2 [label="Small tree 2\\n(fixes errors)"]; t3 [label="... thousands of\\nsmall trees"]; '
               'es [label="Early stopping\\non held-back data"]; '
               's [label="Sum = risk score", fillcolor="#eb6834", fontcolor="white"]; t1 -> t2 -> t3 -> s; '
               'es -> t3 [style=dashed]; }',
    },
}

METRIC_HELP = {
    "Accuracy": "Share of all decisions that are correct. Misleading when defaults are rare: approving everyone is "
                "already very 'accurate'.",
    "Precision": "Of the applicants the model rejects, the share who really would have defaulted.",
    "Recall": "Of the applicants who default, the share the model catches (rejects).",
    "F1": "Harmonic mean of precision and recall: one number that is high only when both are.",
    "LogLoss": "How wrong the scores are when read as probabilities (lower is better). Class weighting inflates the "
               "scores on purpose, so log loss is high even for good rankers.",
    "AUC": "Probability that a random defaulter gets a higher score than a random good applicant (0.5 = guessing, "
           "1 = perfect). Independent of the threshold.",
    "Gini": "2 × AUC − 1: the same ranking quality on the scale credit-risk teams use.",
    "PR_AUC": "Area under the precision-recall curve. Its 'random guess' level equals the default rate, so it shows "
              "how hard the rare defaulters are to find.",
    "KS": "Largest gap between the score distributions of defaulters and good applicants (a classic scorecard "
          "measure).",
}
