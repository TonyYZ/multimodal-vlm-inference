# Run Lyn's unchanged human analysis, then fixed-only VLM performance models.
# --benchmark retains the older human-minus-LLM mixed-model comparison.

####################################
# Files and analysis options
####################################

script_arg <- grep("^--file=", commandArgs(FALSE), value=TRUE)
root <- if (length(script_arg)) {
  dirname(dirname(normalizePath(sub("^--file=", "", script_arg[[1]]))))
} else {
  path <- normalizePath(getwd())
  if (basename(path) == "code") dirname(path) else path
}
args <- commandArgs(TRUE)
study <- if ("--gesture" %in% args) "gesture" else "animation"
sensitivity <- "--sensitivity" %in% args
fixed_vlm <- !sensitivity && !("--benchmark" %in% args)

if (study == "gesture") {
  analysis_dir <- file.path(root, "materials", "gestures", "human_data")
  original_script <- "TSC_EXP_1_Analyses.R"
} else {
  analysis_dir <- file.path(root, "materials", "analyses")
  original_script <- "TSC_EXP_2_Analyses.R"
}
if (study == "gesture" && !fixed_vlm) stop("Gesture analysis currently requires --fixed-vlm.")

output_dir <- file.path(root, "results", study, "human_tests")
if (sensitivity) output_dir <- file.path(output_dir, "sensitivity")
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

####################################
# Original human analysis and plot
####################################

# Run Lyn's script as written, replacing only the Mac-only plotting device.
plot_adapter <- new.env(parent=globalenv())
plot_adapter$device <- NULL
plot_adapter$quartz <- function(width, height) {
  plot_file <- if (sensitivity) NULL else file.path(output_dir, "original_human_bar_chart.pdf")
  pdf(plot_file, width=width, height=height)
  plot_adapter$device <- dev.cur()
}
human <- new.env(parent=plot_adapter)
tryCatch(
  sys.source(file.path(analysis_dir, original_script), envir=human, chdir=TRUE),
  finally=if (!is.null(plot_adapter$device)) dev.off(plot_adapter$device)
)
library(jsonlite)

####################################
# Human likelihood-ratio tests
####################################

# The same likelihood-ratio procedure is used for human and benchmark tests.
likelihood_test <- function(full, term) {
  reduced <- update(full, as.formula(paste(". ~ . -", term)))
  comparison <- anova(full, reduced)
  comparison <- comparison[!is.na(comparison$Chisq), , drop=FALSE]
  stopifnot(nrow(comparison) == 1)
  convergence <- unique(c(full@optinfo$conv$lme4$messages, reduced@optinfo$conv$lme4$messages))
  data.frame(
    coefficient_term=term, estimate=unname(fixef(full)[[term]]),
    std_error=unname(sqrt(diag(vcov(full)))[[term]]),
    chi_square=comparison$Chisq, df=comparison$Df,
    p_value=comparison[["Pr(>Chisq)"]], n_observations=nobs(full),
    singular_fit=isSingular(full) || isSingular(reduced),
    convergence_message=paste(convergence, collapse="; ")
  )
}

# Export Lyn's final human models without changing or refitting their full models.
diagnostics <- data.frame(
  dataset=c("dsi", "isi", "q", "none", "supp", "positive", "negative"),
  condition=c("Scalar Implicature", "Scalar Implicature", "Presupposition",
              "Presupposition", "Supplement", "Homogeneity", "Homogeneity"),
  environment=c("Positive", "Negative", "Question", "None", "Unembedded", "Positive", "Negative")
)
human_results <- list()
for (i in seq_len(nrow(diagnostics))) {
  diagnostic <- diagnostics[i, ]
  data <- human[[diagnostic$dataset]]
  full <- human[[paste0(diagnostic$dataset, ".lm")]]
  if (diagnostic$dataset %in% c("dsi", "isi", "supp")) {
    interaction <- if (study == "gesture") "GestureC:InferenceC" else "InferenceC:GestureC"
    tests <- c(Premise="GestureC", Inference="InferenceC", `Premise x inference`=interaction)
  } else {
    inference <- if (diagnostic$condition == "Homogeneity") "InferenceTypeC" else "InferenceC"
    tests <- c(Inference=inference)
  }
  for (label in names(tests)) {
    human_results[[length(human_results) + 1]] <- cbind(
      condition=diagnostic$condition, environment=diagnostic$environment,
      tested_effect=label, likelihood_test(full, tests[[label]]),
      n_participants=length(unique(data$SubjID)),
      n_stories=length(unique(if (study == "gesture") data$Group else data$Story))
    )
  }
}
if (!sensitivity) {
  write.csv(do.call(rbind, human_results),
            file.path(output_dir, "original_human_likelihood_ratio_tests.csv"), row.names=FALSE)
}

####################################
# VLM models
####################################

if (fixed_vlm) {
  source(file.path(root, "code", "vlm_fixed_effects.R"))
} else {
  if (sensitivity) source(file.path(root, "code", "random_effect_sensitivity.R"))
  source(file.path(root, "code", "vlm_benchmark_tests.R"))
}
