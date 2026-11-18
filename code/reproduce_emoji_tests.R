# Execute the original human models with local paths, then their fixed-only VLM counterparts.
arg <- grep("^--file=", commandArgs(FALSE), value=TRUE)
root <- if (length(arg)) dirname(dirname(normalizePath(sub("^--file=", "", arg[1])))) else normalizePath(getwd())
source_dir <- file.path(root, "materials", "emoji_study", "source_cache")
output <- file.path(root, "results", "emoji", "human_tests")
dir.create(output, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(file.path(root, "tmp", "R-library"), .libPaths()))
library(lme4)
allow_incomplete <- "--allow-incomplete" %in% commandArgs(TRUE)
if (!requireNamespace("lmtest", quietly=TRUE)) stop("Install lmtest for the VLM ML likelihood-ratio tests.")

spec <- rbind(
  data.frame(experiment="Experiment 1", dataset=c("eachexi", "eachuni", "each", "exactexi", "exactuni", "exact", "might", "neg", "noneexi", "noneuni", "none", "unem"),
    environment=c("Each", "Each", "Each", "Exactly-one", "Exactly-one", "Exactly-one", "Might", "Negation", "None", "None", "None", "Unembedded"),
    reading=c("Existential", "Universal", "Interaction", "Existential", "Universal", "Interaction", "Conditional", "Conditional", "Existential", "Universal", "Interaction", "Manner")),
  data.frame(experiment="Experiment 2", dataset=c("data", "unem", "q", "might", "neg", "none"),
    environment=c("All environments", "Unembedded", "Question", "Might", "Negation", "None"), reading="Inference"),
  data.frame(experiment="Experiment 3", dataset="post", environment="All emojis", reading="Inference")
)
spec$test <- ifelse(spec$experiment == "Experiment 1", paste("Co-text:", spec$environment, "/", spec$reading),
                    ifelse(spec$experiment == "Experiment 2", paste("Pro-text:", spec$environment), "Post-text: All emojis"))
spec$term <- ifelse(spec$reading == "Interaction", "ConditionC:ReadingC", "ConditionC")
ratings <- read.csv(file.path(root, "results/emoji/inference/comparison/emoji_pilot_model_human.csv"), check.names=FALSE)
manifest <- read.csv(file.path(root, "materials/emoji_study/emoji_pilot_manifest.csv"), check.names=FALSE)

select_cells <- function(data, s) {
  data <- subset(data, experiment == s$experiment)
  if (!s$environment %in% c("All environments", "All emojis")) data <- subset(data, environment == s$environment)
  if (s$experiment == "Experiment 1" && s$reading != "Interaction") data <- subset(data, inference_type == s$reading)
  data
}

test_effect <- function(full, term, reduced=NULL, expected_n=nobs(full)) {
  mixed <- inherits(full, "merMod")
  beta <- if (mixed) fixef(full) else coef(full)
  result <- data.frame(estimate=unname(beta[[term]]), std_error=NA_real_, chi_square=NA_real_, df=NA_real_,
    p_value=NA_real_, log_likelihood_full=NA_real_, log_likelihood_reduced=NA_real_,
    n_observations=nobs(full), n_expected=expected_n, missing_cells=expected_n-nobs(full),
    residual_df=if (mixed) NA_real_ else df.residual(full),
    singular_fit=if (mixed) isSingular(full) else NA, analysis_status="ok", convergence_message="")
  if (nobs(full) != expected_n) {
    result$estimate <- NA_real_
    result$analysis_status <- "incomplete input cells; coefficient withheld"
    return(result)
  }
  if (!mixed && (df.residual(full) <= 0 || deviance(full) <= 1e-10 * max(1, sum(model.response(model.frame(full))^2)))) {
    result$analysis_status <- if (df.residual(full) <= 0) "no residual degrees of freedom" else "zero residual variance"
    return(result)
  }
  if (is.null(reduced)) {
    fit_data <- model.frame(full)
    reduced <- update(full, as.formula(paste(". ~ . -", term)), data=fit_data)
  }
  comparison <- if (mixed) anova(full, reduced) else lmtest::lrtest(reduced, full)
  row <- which(!is.na(comparison$Chisq))
  stopifnot(length(row) == 1)
  result$std_error <- unname(sqrt(diag(vcov(full)))[[term]])
  result$chi_square <- comparison$Chisq[row]
  result$df <- abs(comparison$Df[row])
  result$p_value <- comparison[["Pr(>Chisq)"]][row]
  result$log_likelihood_full <- as.numeric(logLik(full))
  result$log_likelihood_reduced <- as.numeric(logLik(reduced))
  if (mixed) {
    result$singular_fit <- isSingular(full) || isSingular(reduced)
    result$convergence_message <- paste(unique(c(full@optinfo$conv$lme4$messages, reduced@optinfo$conv$lme4$messages)), collapse="; ")
  }
  result
}

results <- list()
for (experiment in 1:3) {
  adapter <- new.env(parent=globalenv())
  adapter$length.unique <- function(x) length(unique(x))
  paths <- c("target.csv"="exp1_target.csv", "control.csv"="exp1_control.csv", "protext.csv"="exp2_protext.csv", "task.csv"="exp3_task.csv")
  adapter$read.csv <- function(file, ...) utils::read.csv(file.path(source_dir, paths[[file]]), ...)
  adapter$quartz <- function(width, height) pdf(file.path(output, paste0("exp", experiment, "_original_human_bar_chart.pdf")), width=width, height=height)
  human <- new.env(parent=adapter)
  lines <- readLines(file.path(source_dir, paste0("exp", experiment, "_analysis.R")), warn=FALSE)
  # Demographics files are unavailable; this section reports counts, not exclusions.
  first <- grep('^header <- names\\(read.csv\\("(target|protext|task)\\.csv"', lines)[1]
  lines <- c(lines[seq_len(grep("# Demographic data", lines)[1] - 2)], lines[first:length(lines)])
  # The distributed Experiment 3 script uses `post` without defining it.
  if (experiment == 3) lines <- append(lines, 'post <- subset(data, Emoji.Type == "Post")', after=grep('^post\\$ConditionC', lines)[1] - 1)
  device <- dev.cur()
  tryCatch(eval(parse(text=lines), envir=human), finally=if (dev.cur() != device) dev.off())
  for (i in which(spec$experiment == paste("Experiment", experiment))) {
    s <- spec[i, ]
    h <- human[[s$dataset]]
    full <- human[[paste0(s$dataset, ".lm")]]
    reduced <- human[[paste0(s$dataset, if (s$reading == "Interaction") ".lm_interaction" else ".lm_ConditionC")]]
    prefix <- data.frame(model="Human", input_format="human", test=s$test, test_order=i,
      experiment=s$experiment, environment=s$environment, reading=s$reading, coefficient_term=s$term)
    results[[length(results)+1]] <- cbind(prefix, test_effect(full, s$term, reduced),
      n_items=length(unique(h$Emoji)), n_participants=length(unique(h$Participant.Private.ID)),
      model_formula=paste(deparse(formula(full)), collapse=" "), test_design="Original human mixed-effects ML LRT")
    expected <- select_cells(manifest, s)
    for (model in unique(ratings$model)) {
      d <- select_cells(ratings[ratings$model == model, ], s)
      stopifnot(nrow(d) <= nrow(expected), !anyDuplicated(d[c("source_video", "statement_type")]))
      if (nrow(d) != nrow(expected) && !allow_incomplete)
        stop("Missing cells for ", model, ": ", s$test, ". Inspect responses; --allow-incomplete withholds affected coefficients.")
      d$Response <- d$model_score
      stopifnot(all(is.finite(d$Response)), all(d$Response >= 0 & d$Response <= 100))
      d$ConditionC <- if (experiment == 1) ifelse(d$material_condition == "Target", -.5, .5) else ifelse(d$statement_type == "target", -.5, .5)
      d$ReadingC <- ifelse(d$inference_type == "Existential", -.5, .5)
      fixed <- if (s$reading == "Interaction") Response ~ ConditionC * ReadingC else Response ~ ConditionC
      prefix$model <- model
      prefix$input_format <- "pure-text"
      results[[length(results)+1]] <- cbind(prefix, test_effect(lm(fixed, data=d), s$term, expected_n=nrow(expected)),
        n_items=length(unique(d$emoji)), n_participants=0,
        model_formula=paste(deparse(fixed), collapse=" "), test_design="Fixed-only Gaussian ML LRT (lmtest::lrtest)")
    }
  }
}
write.csv(do.call(rbind, results), file.path(output, "emoji_fixed_effect_tests.csv"), row.names=FALSE)
writeLines(c("Original scripts unchanged; local data paths mapped by read.csv adapter.",
  "Demographic-only reporting skipped: source files unavailable; no eligibility exclusions changed.",
  "Experiment 3 repair: define post = subset(data, Emoji.Type == 'Post').",
  "Human fits use full original items. VLM item counts are saved per fit; verify matching coverage before interpretation.",
  "--allow-incomplete withholds coefficients for incomplete input cells; no missing responses are imputed.",
  "New explicit post-text premise controls have no human data and are excluded from these original-script tests.",
  "No LLM participants or replicated responses invented; saturated/zero-variance LRTs stay unavailable."),
  file.path(output, "reproduction_notes.txt"))
cat("Wrote original human and fixed-only VLM emoji coefficients/LRTs to", output, "\n")
