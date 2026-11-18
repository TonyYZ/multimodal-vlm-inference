# Optional human-minus-VLM comparison, called by reproduce_original_human_tests.R.

# Each human response retains its SubjID and receives its matching fixed LLM benchmark.
# Reusing that benchmark does not create additional LLM observations or participants.
keys <- c("Condition", "Environment", "Story", "GesturalPremise", "InferenceType")
video_cells <- unique(human$df[keys])
video_cells$source_video <- paste0(
  gsub(" ", "", video_cells$Condition), video_cells$Story, "_",
  video_cells$GesturalPremise, "Premise",
  ifelse(video_cells$Environment == "Unembedded", "",
         paste0("(", video_cells$Environment, ")")), ".m4v"
)
video_cells$statement_type <- tolower(as.character(video_cells$InferenceType))
design_warning <- paste(
  "Individual human ratings; LLM scores are fixed benchmarks for the tested materials.",
  "Uncertainty is over humans, not LLM runs; only 2-3 stories. Inspect fit warnings."
)
message(design_warning)

read_ratings <- function(path) {
  connection <- file(path, open="r", encoding="UTF-8")
  tryCatch(stream_in(connection, verbose=FALSE), finally=close(connection))
}

parse_score <- function(response) {
  # Strip Gemma's standalone marker, not arbitrary prose; preserve the raw response.
  text <- sub("^\\s*thought[ \\t]*\\r?\\n", "", as.character(response), ignore.case=TRUE, perl=TRUE)
  score <- strcapture("^\\s*([0-9]{1,3}(?:\\.[0-9]+)?)(?:\\s|$)", text,
                      proto=list(value=double()), perl=TRUE)$value
  score[!is.finite(score) | score > 100] <- NA_real_
  score
}

rating_dir <- file.path(root, "results", "animation", "inference", "ratings")
rating_files <- list.files(rating_dir, pattern="_sequence_ratings\\.jsonl$", full.names=TRUE)
rating_files <- rating_files[!grepl("_presupposition_positive_", basename(rating_files))]
if (!length(rating_files)) stop("No model rating files found in ", rating_dir)
benchmark_results <- list()

for (path in rating_files) {
  ratings <- read_ratings(path)
  model <- unique(ratings$model)
  input_format <- unique(ratings$input_mode)
  stopifnot(length(model) == 1, length(input_format) == 1)
  ratings$source_video <- basename(gsub("\\\\", "/", ratings$source_video))
  ratings$statement_type <- tolower(ratings$statement_type)
  ratings$llm_value <- parse_score(ratings$response)
  paired <- merge(
    video_cells, ratings[c("source_video", "statement_type", "llm_value")],
    by=c("source_video", "statement_type")
  )

  for (i in seq_len(nrow(diagnostics))) {
    diagnostic <- diagnostics[i, ]
    h <- human[[diagnostic$dataset]]
    m <- subset(paired, Condition == diagnostic$condition & Environment == diagnostic$environment)
    three_way <- diagnostic$condition %in% c("Scalar Implicature", "Supplement")
    template <- human[[paste0(diagnostic$dataset, ".lm")]]
    term <- if (three_way) "InferenceC:GestureC" else {
      if (diagnostic$condition == "Homogeneity") "InferenceTypeC" else "InferenceC"
    }
    benchmark_formula <- update.formula(formula(template), discrepancy ~ .)
    llm_coefficient <- NA_real_
    result <- tryCatch({
      if (nrow(m) != nrow(unique(h[keys])) || anyDuplicated(m[keys]) || anyNA(m$llm_value)) {
        stop("Incomplete, duplicate, or unscored model cells; diagnostic not fitted.")
      }
      # LLM-only behavioral coefficients: same contrasts, Story blocking, no SubjID.
      m$GestureC <- ifelse(m$GesturalPremise == "Target", -0.5, 0.5)
      m$InferenceC <- ifelse(m$InferenceType == "Target", -0.5, 0.5)
      llm_formula <- if (three_way) {
        llm_value ~ InferenceC * GestureC + factor(Story)
      } else llm_value ~ InferenceC + factor(Story)
      llm_term <- if (three_way) "InferenceC:GestureC" else "InferenceC"
      llm_coefficient <- unname(coef(lm(llm_formula, data=m))[[llm_term]])
      combined <- merge(h, m[c(keys, "llm_value")], by=keys, all.x=TRUE)
      stopifnot(nrow(combined) == nrow(h), !anyNA(combined$llm_value))
      combined$discrepancy <- combined$value - combined$llm_value
      if (sensitivity) {
        sensitivity_tests(benchmark_formula, combined, term)
      } else {
        full <- lmer(benchmark_formula, REML=FALSE, data=combined,
                     control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000)))
        cbind(likelihood_test(full, term), analysis_status="ok")
      }
    }, error=function(error) {
      if (sensitivity) sensitivity_tests(benchmark_formula, NULL, term, conditionMessage(error)) else data.frame(
        coefficient_term=term, estimate=NA_real_, std_error=NA_real_,
        chi_square=NA_real_, df=NA_real_, p_value=NA_real_, n_observations=0L,
        singular_fit=NA, convergence_message="", analysis_status=conditionMessage(error)
      )
    })
    benchmark_results[[length(benchmark_results) + 1]] <- cbind(
      model=model, input_format=input_format, condition=diagnostic$condition,
      environment=diagnostic$environment, diagnostic_term=term, result,
      n_participants=length(unique(h$SubjID)), n_stories=length(unique(h$Story)),
      n_llm_scores=nrow(m), optimizer="bobyqa",
      model_formula=if (sensitivity) result$candidate_formula else paste(deparse(benchmark_formula), collapse=" "),
      human_diagnostic_coefficient=unname(fixef(template)[[term]]),
      llm_diagnostic_coefficient=llm_coefficient,
      human_aggregation="none; individual ratings", design_warning=design_warning
    )
  }
  if (sensitivity) cat("Sensitivity completed:", model, "[", input_format, "]\n")
}
write.csv(do.call(rbind, benchmark_results),
          file.path(output_dir, if (sensitivity) "random_effects_tests.csv" else "llm_fixed_benchmark_tests.csv"), row.names=FALSE)
cat("Wrote human and fixed-benchmark tests to", output_dir, "\n")
