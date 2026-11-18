# Lyn's likelihood-ratio tests (materials/analyses/TSC_EXP_2_Analyses.R) applied to
# temperature-sampled model ratings (animation study).
#
# Each model's 99 temperature samples are treated as 99 pseudo-participants
# (sample k across all 48 statements = participant k), so every source is analysed like Lyn's
# 99 human participants. Two analyses per test:
#   within:           Lyn's model fitted to one source alone
#                     -> does this source (humans or a model) show the effect?
#   human comparison: humans + one model pooled, with SourceC (-.5 = human, +.5 = model);
#                     LRT on the highest interaction with SourceC
#                     -> does the model's effect differ from the human effect?
#
# Coding as in Lyn's script: InferenceC = -.5 Target / +.5 Baseline,
#                            PremiseC   = -.5 Target / +.5 Control premise animation
#                            (called GestureC in Lyn's script; the name was carried over from the gesture study).
# All models are fitted with ML (REML=FALSE) and compared with anova(full, reduced).
#
# Random effects: each test lists Lyn's final structure first, followed by simpler ones. If the full
# or any reduced model fails to converge, the next structure in the list is used. Singular fits are
# kept, as in Lyn's analyses.
#
# Effect on the d scale (the effect plotted in the paper):
#   two-factor tests: d = (TT - TB) - (CT - CB) = coefficient of InferenceC:PremiseC
#   one-factor tests: d = Target - Baseline     = -coefficient of InferenceC
#   In the comparison models the same transformation of the SourceC interaction gives d_model - d_human.
#
# Input:  results/animation/sampling_pilot/pseudo_participant_ratings.csv (from sampled_human_comparison.py)
# Output: results/animation/sampling_pilot/sampled_likelihood_ratio_tests.csv
# Run (Oberon): pixi run --manifest-path /store/work/users/yzhou/r-lme4/pixi.toml Rscript code/sampled_lrt.R
#
# Sensitivity: `Rscript code/sampled_lrt.R --maximal` starts every test from the maximal structure
# (also SI positive and Presupposition question, where Lyn's final model is simpler) and writes
# sampled_likelihood_ratio_tests_maximal.csv.

suppressPackageStartupMessages(library(lme4))


####################################
# Prepare data
####################################

script_arg <- grep("^--file=", commandArgs(FALSE), value=TRUE)
root <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg[[1]]))))
out_dir <- file.path(root, "results", "animation", "sampling_pilot")
out_arg <- grep("^--out=", commandArgs(TRUE), value=TRUE)  # optional: --out=<folder relative to the repo>
if (length(out_arg)) out_dir <- file.path(root, sub("^--out=", "", out_arg))
maximal <- "--maximal" %in% commandArgs(TRUE)

df <- read.csv(file.path(out_dir, "pseudo_participant_ratings.csv"), na.strings="")
df$InferenceC <- ifelse(df$InferenceType=="Target", -.5, .5)
df$PremiseC <- ifelse(df$GesturalPremise=="Target", -.5, .5)
df$SourceC <- ifelse(df$source=="Human", -.5, .5)
df$Story <- factor(df$Story)

models <- setdiff(unique(df$source), "Human")


####################################
# Helpers
####################################

quiet <- function(x) suppressWarnings(suppressMessages(x))

# Singular fits count as converged; only optimizer failures trigger a simpler structure.
converged <- function(fit) all(grepl("singular", fit@optinfo$conv$lme4$messages))

# Fits `fixed + random` with the first random structure under which the full model and every
# reduced model (full minus one tested term) converge. Returns one likelihood-ratio test per term.
lrt <- function(data, fixed, randoms, terms, d_sign) {
  for (random in randoms) {
    full <- quiet(lmer(as.formula(paste(fixed, "+", random)), REML=FALSE, data=data))
    reduced <- lapply(terms, function(term) quiet(update(full, as.formula(paste(". ~ . -", term)))))
    ok <- converged(full) && all(sapply(reduced, converged))
    if (ok) break
  }
  do.call(rbind, lapply(seq_along(terms), function(i) {
    comparison <- anova(full, reduced[[i]])
    beta <- unname(fixef(full)[[terms[[i]]]])
    data.frame(tested_effect=names(terms)[[i]], coefficient_term=terms[[i]],
               estimate=beta, d_scale_estimate=d_sign * beta,
               chi_square=comparison$Chisq[2], df=comparison$Df[2], p_value=comparison[["Pr(>Chisq)"]][2],
               random_effects=random, converged=ok,
               singular_fit=isSingular(full) || any(sapply(reduced, isSingular)),
               n_observations=nobs(full))
  }))
}

# Runs one test: the within model on humans and on every model, then each model against humans.
run_test <- function(condition, environment, within, comparison, randoms,
                     within_terms, comparison_term, d_sign) {
  data <- df[df$condition==condition & df$environment==environment, ]
  rows <- list()
  for (source in c("Human", models)) {
    rows[[length(rows) + 1]] <- data.frame(
      condition=condition, environment=environment, model=source, analysis="within",
      lrt(data[data$source==source, ], within, randoms, within_terms, d_sign))
    if (source=="Human") next
    rows[[length(rows) + 1]] <- data.frame(
      condition=condition, environment=environment, model=source, analysis="human comparison",
      lrt(data[data$source %in% c("Human", source), ], comparison, randoms, comparison_term, d_sign))
    cat(condition, environment, "|", source, "\n")
  }
  do.call(rbind, rows)
}

# Tested terms of the two kinds of model (names are the labels in the output).
two_factor_terms <- c("Premise"="PremiseC", "Inference"="InferenceC", "Premise x inference"="InferenceC:PremiseC")
one_factor_terms <- c("Inference"="InferenceC")

results <- list()


####################################
# Models
####################################

#===================================
# Scalar implicatures
#===================================

# Scalar implicatures (positive)
# Lyn's maximal model failed to converge; her final model drops the interaction slope and (1|Story).
results$si_positive <- run_test(
  "Scalar Implicature", "Positive",
  within     = "value ~ InferenceC*PremiseC",
  comparison = "value ~ InferenceC*PremiseC*SourceC",
  randoms    = c(if (maximal) c("(1+InferenceC*PremiseC|participant)+(1|Story)",
                                "(1+InferenceC*PremiseC|participant)"),
                 "(1+InferenceC+PremiseC|participant)",
                 "(1|participant)+(1|Story)",
                 "(1|participant)"),
  within_terms = two_factor_terms,
  comparison_term = c("Effect x source"="InferenceC:PremiseC:SourceC"),
  d_sign = 1)

# Scalar implicatures (negative)
results$si_negative <- run_test(
  "Scalar Implicature", "Negative",
  within     = "value ~ InferenceC*PremiseC",
  comparison = "value ~ InferenceC*PremiseC*SourceC",
  randoms    = c("(1+InferenceC*PremiseC|participant)+(1|Story)",
                 "(1+InferenceC*PremiseC|participant)",
                 "(1+InferenceC+PremiseC|participant)",
                 "(1|participant)+(1|Story)",
                 "(1|participant)"),
  within_terms = two_factor_terms,
  comparison_term = c("Effect x source"="InferenceC:PremiseC:SourceC"),
  d_sign = 1)

#===================================
# Presuppositions
#===================================

# Presuppositions (questions)
# Lyn's final model keeps only by-participant intercepts (her richer models failed to converge).
results$presupposition_question <- run_test(
  "Presupposition", "Question",
  within     = "value ~ InferenceC",
  comparison = "value ~ InferenceC*SourceC",
  randoms    = c(if (maximal) c("(1+InferenceC|participant)+(1|Story)",
                                "(1+InferenceC|participant)",
                                "(1|participant)+(1|Story)"),
                 "(1|participant)"),
  within_terms = one_factor_terms,
  comparison_term = c("Effect x source"="InferenceC:SourceC"),
  d_sign = -1)

# Presuppositions ("none")
results$presupposition_none <- run_test(
  "Presupposition", "None",
  within     = "value ~ InferenceC",
  comparison = "value ~ InferenceC*SourceC",
  randoms    = c("(1+InferenceC|participant)+(1|Story)",
                 "(1+InferenceC|participant)",
                 "(1|participant)+(1|Story)",
                 "(1|participant)"),
  within_terms = one_factor_terms,
  comparison_term = c("Effect x source"="InferenceC:SourceC"),
  d_sign = -1)

#===================================
# Supplements
#===================================

results$supplement <- run_test(
  "Supplement", "Unembedded",
  within     = "value ~ InferenceC*PremiseC",
  comparison = "value ~ InferenceC*PremiseC*SourceC",
  randoms    = c("(1+InferenceC*PremiseC|participant)+(1|Story)",
                 "(1+InferenceC*PremiseC|participant)",
                 "(1+InferenceC+PremiseC|participant)",
                 "(1|participant)+(1|Story)",
                 "(1|participant)"),
  within_terms = two_factor_terms,
  comparison_term = c("Effect x source"="InferenceC:PremiseC:SourceC"),
  d_sign = 1)

#===================================
# Homogeneity
#===================================

# Homogeneity (positive)
results$homogeneity_positive <- run_test(
  "Homogeneity", "Positive",
  within     = "value ~ InferenceC",
  comparison = "value ~ InferenceC*SourceC",
  randoms    = c("(1+InferenceC|participant)+(1|Story)",
                 "(1+InferenceC|participant)",
                 "(1|participant)+(1|Story)",
                 "(1|participant)"),
  within_terms = one_factor_terms,
  comparison_term = c("Effect x source"="InferenceC:SourceC"),
  d_sign = -1)

# Homogeneity (negative)
results$homogeneity_negative <- run_test(
  "Homogeneity", "Negative",
  within     = "value ~ InferenceC",
  comparison = "value ~ InferenceC*SourceC",
  randoms    = c("(1+InferenceC|participant)+(1|Story)",
                 "(1+InferenceC|participant)",
                 "(1|participant)+(1|Story)",
                 "(1|participant)"),
  within_terms = one_factor_terms,
  comparison_term = c("Effect x source"="InferenceC:SourceC"),
  d_sign = -1)


####################################
# Save
####################################

results <- do.call(rbind, unname(results))
write.csv(results, file.path(out_dir, paste0("sampled_likelihood_ratio_tests", if (maximal) "_maximal", ".csv")),
          row.names=FALSE)
cat("Wrote", nrow(results), "tests\n")
